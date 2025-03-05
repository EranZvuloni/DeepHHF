import numpy as np
import torch
from model_components import DeepHHFModel
from preprocess_components import Preprocessor
from explainability import AttentionGradRollout, explain_plot

# Load an example:
example_ind = 4623
# example_ind = 15740
# example_ind = 33217
# example_ind = 24625
signal = np.load(f'data/Holter_{example_ind}.npz')['signal']

# Preprocess:
signal = Preprocessor()(signal)

# Load the DeepHHF model:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_parameters = {'encoder_only': False,
                    'compact_strides': [2, 4, 6, 8], 'compact_filters': 32, 'compact_kernel': 3,
                    'd_model': 512, 'nhead': 16, 'num_layers': 4, 'dim_feedforward': 2048,
                    'dropout_p': 0.15000000000000002,
                    }

model = DeepHHFModel(input_shape=signal.shape, init_device=device, **model_parameters).to(device)

model.load_state_dict(torch.load('model/model_state.pth', map_location=device))

if device.type == 'cpu':  # The model was trained with dtype=torch.bfloat16
    model = model.to(torch.torch.float32)
    signal = signal.to(torch.torch.float32)
else:
    model = model.to(torch.torch.bfloat16)
    signal = signal.to(torch.torch.bfloat16).to(device)

model.eval()

# Predict:
with torch.no_grad():
    model_output = torch.nn.Sigmoid()(model(signal)).item()

# Thresholds computed externally based on experimental results:
moderate_th = 0.50390625
high_th = 0.8046875

if model_output > high_th:
    risk = 'HIGH'
elif model_output > moderate_th:
    risk = 'MODERATE'
else:
    risk = 'LOW'

print(f"Patient risk level to develop heart failure in the next 5 years is {risk}; model output probability={round(model_output, 3)}")

# Explainability analysis:
att_grad_rollout = AttentionGradRollout(model, attention_layer_name='attn', discard_ratio=0.9)
mask, A_map = att_grad_rollout(signal, by_heads=False)
daytime_start = np.load('data/daytime_starts.npy', allow_pickle=True).item()[example_ind]  # Load the clock time of the recording
explain_plot(signal, daytime_start, mask)  # will plot using plt.show()
