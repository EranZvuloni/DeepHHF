import numpy as np
import torch
import matplotlib.pyplot as plt


def grad_rollout(attentions, gradients, discard_ratio):
    """
    AttentionGradRollout: Adapted from Jacob Gildenblat (jacobgil.github.io) to match the Holter ECG signal analysis.
    """

    result = torch.eye(attentions[0][0].size(-1))
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            attention_heads_fused = (attention*weights).mean(axis=1)  # mean over head dim.
            attention_heads_fused[attention_heads_fused < 0] = 0

            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)
            result = torch.matmul(a, result)

    A = result.squeeze().numpy()  # Return the composite attention map as well.
    mask = result[0, 0, :].numpy()  # Here there is no selection of the class token as in ViT.
    # mask = result[0, :, :].numpy().mean(axis=0)  # Another option is to take the mean.
    mask = mask / np.max(mask)
    return mask, A


def rollout_by_heads(attentions, gradients, discard_ratio):
    n_heads = attentions[0].shape[1]
    masks = np.zeros((n_heads, attentions[0][0].size(-1)))
    As = np.zeros((n_heads, attentions[0][0].size(-2), attentions[0][0].size(-1)))
    for h in range(n_heads):
        h_attentions = [attention[:, h:h + 1, :, :] for attention in attentions]
        h_gradients = [gradient[:, h:h + 1, :, :] for gradient in gradients]
        h_mask, h_A = grad_rollout(h_attentions, h_gradients, discard_ratio)
        masks[h, :] = h_mask
        As[h, :, :] = h_A
    return masks, As


class AttentionGradRollout:
    def __init__(self, model, attention_layer_name, discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self.hook_handles = []
        self.handle_counter = 0
        for name, module in self.model.named_modules():
            if (attention_layer_name in name) and not ('proj' in name):
                handle = module.register_forward_hook(self.get_attention)
                self.hook_handles.append(handle)  # get the handles to later remove the hooks.

                # The gradients will be extracted by in the forward hook manually,
                # since they do not flow in the regular gradient flow
                # (because the map is the secondary output of the MultiHeadAttention layer,
                # and its gradients are not retained in its forward method).

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        # To extract the attention maps from the model, remove hooks to enable forward through
        # the model without endless recursive.
        self.hook_handles[self.handle_counter].remove()
        self.handle_counter += 1

        # Then, manually use the attention map extracted form the MultiHeadAttention to forward and backward,
        # and get the map nad its gradients:
        Q = input[0].clone().detach().requires_grad_(True)
        K = input[1].clone().detach().requires_grad_(True)
        V = input[2].clone().detach().requires_grad_(True)

        _, A = module(Q, K, V, average_attn_weights=False)  # second element is the attention map - batchXheadsXseqXseq
        A.retain_grad()
        out = A@V
        L = out.sum()
        L.backward()
        G = A.grad
        self.attentions.append(A.cpu())
        self.attention_gradients.append(G.cpu())

    def __call__(self, input_tensor, category_index=None, by_heads=False):
        self.model.zero_grad()
        output = self.model(input_tensor)
        # We don't have categories, since there is no multiple classes and class token.
        # We don't need to backward because the forward hook already extracts what is needed.
        if by_heads:
            return rollout_by_heads(self.attentions, self.attention_gradients, self.discard_ratio)
        else:
            return grad_rollout(self.attentions, self.attention_gradients, self.discard_ratio)


def explain_plot(signal, daytime_start, mask, fs=128):
    mask = np.repeat(mask, repeats=30 * fs, axis=-1)  # 30 second one window duration.

    signal = signal.squeeze().to(float).cpu().numpy()
    t = np.linspace(0, 24, len(signal))
    mask_signal = mask * signal.max()

    # Use the day-time start string to align the recording to the clock time.
    if len(daytime_start) == 4:
        daytime_start = '0' + daytime_start
    daytime_start_sec = int(daytime_start[:2]) * 60 * 60 + int(daytime_start[3:]) * 60
    daytime_start_sample = int(daytime_start_sec * 128 // (24 / 6))  # 6 is the segmented signal in hours

    plt.figure(figsize=(20, 6))
    rolled_signal = np.roll(signal, daytime_start_sample)
    rolled_mask_signal = np.roll(mask_signal, daytime_start_sample)
    plt.plot(t, rolled_signal, color='darkviolet', label='Segmented signal')
    plt.plot(t, rolled_mask_signal, linewidth=3, color='lawngreen', label='Gradient attention rollout')
    plt.xlabel('Time (hours after midnight)', fontsize=14)
    plt.ylabel('ECG (mV)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim((0, 24))
    plt.legend(loc="lower right", fontsize=14)
    plt.tight_layout()
    plt.show()
