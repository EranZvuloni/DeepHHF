import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    # Copied as is from the PyTorch tutorial

    The batch first option was added.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        if self.batch_first:  # added
            pe = pe.transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ''[seq_len, batch_size, embedding_dim]''
        """
        if self.batch_first:
            x = x + self.pe[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0]]
        return self.dropout(x)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=17, dropout_p=0.0, pool_size=1):
        super().__init__()
        self.direct_path = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),  # padding for "same" settings.
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                      stride=pool_size,  # stride is adequate to the pool in the skip path
                      padding=kernel_size // 2)
        )

        skip_layers = []
        if in_channels != out_channels:
            skip_layers.append(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, stride=pool_size, kernel_size=1))  # adapt depth
        self.skip_path = nn.Sequential(*skip_layers)

        self.merged_path = nn.Sequential(
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        direct = self.direct_path(x)
        skip = self.skip_path(x)

        if skip.shape[-1] < direct.shape[-1]:  # pad skip in case of odd dimension
            skip = nn.functional.pad(skip, (0, 1))

        merged = self.merged_path(direct + skip)
        return merged


class ResAndStride(nn.Module):
    def __init__(self, filters, stride_kernel, res_kernel=3, dropout_p=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            ResBlock1D(in_channels=filters, out_channels=filters, kernel_size=res_kernel, dropout_p=dropout_p, pool_size=1),
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=stride_kernel * 2, stride=stride_kernel),
            nn.BatchNorm1d(num_features=filters),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class CompactEncoder(nn.Module):
    """
    Inspired by Defossez et al.; EnCodec: High fidelity Neural Audio Compression
    """
    def __init__(self, input_shape, stride_kernels: list, filters=32, first_conv_kernel=7, dropout_p=0):
        super().__init__()

        # Add first convolutional layer:
        layers = [
            nn.Sequential(
                nn.Conv1d(in_channels=input_shape[1], out_channels=filters, kernel_size=first_conv_kernel, stride=1),
                nn.BatchNorm1d(num_features=filters),
                nn.ReLU(),
                nn.Dropout(p=dropout_p)
            )
        ]

        # Add residual blocks with strides layers afterward:
        for stride_kernel in stride_kernels:
            layers.append(ResAndStride(filters, stride_kernel, dropout_p=dropout_p))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class CompactClassifier(nn.Module):
    def __init__(self, classifier_in, class_hidden, n_classes):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=classifier_in, out_features=class_hidden),
            nn.BatchNorm1d(class_hidden),
            nn.ReLU(),
            nn.Linear(in_features=class_hidden, out_features=n_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class TransformerHead(nn.Module):
    def __init__(self, input_features, d_model=None, nhead=4, num_layers=1, dim_feedforward=None):
        super().__init__()

        if d_model is None:
            d_model = input_features

        if dim_feedforward is None:
            dim_feedforward = d_model

        layers = []
        # Set a linear transformation if d_model is different from the input:
        if input_features != d_model:
            layers += [nn.Linear(input_features, d_model)]

        # Add the Transformer:
        layers += [PositionalEncoding(d_model=d_model, dropout=0, batch_first=True)]
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0, batch_first=True)
        layers += [nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x_t = self.layers(x)
        return x_t, None


class DeepHHFModel(nn.Module):
    def __init__(self, input_shape, init_device, n_classes=1,
                 signal_fs=128, window_length=0.5,  # Input parameters; window_length in minutes
                 encoder_only=False, compact_strides=None, compact_filters=None, compact_kernel=None, # Encoder parameters
                 d_model=128, nhead=4, num_layers=1, dim_feedforward=None,  # Transformer parameters
                 dropout_p=0.0,  # Global parameters
                 ):
        super().__init__()
        self.fs = signal_fs
        self.window_length = window_length
        self.encoder_only = encoder_only

        # Define an input to forward during initialization for computing all dimensions:
        dummy_batch_size = 1
        zero_input = torch.zeros((dummy_batch_size, input_shape[1], input_shape[2])).to(init_device)
        zero_windows = self.input2windows(zero_input)

        # Define encoder:
        self.encoder = CompactEncoder(input_shape,
                                      stride_kernels=compact_strides,
                                      filters=compact_filters, first_conv_kernel=compact_kernel,
                                      dropout_p=dropout_p).to(init_device)
        # Forward through encoder:
        with torch.no_grad():
            zero_z1 = self.encoder(zero_windows)

        if not encoder_only:
            # Prepare for GRU\Transformer:
            zero_z1_flat = self.encoder2gru(zero_z1, dummy_batch_size)

            # Define GRU\Transformer:
            self.gru_layer = TransformerHead(input_features=zero_z1_flat.shape[2],
                                             d_model=d_model,
                                             nhead=nhead,
                                             num_layers=num_layers,
                                             dim_feedforward=dim_feedforward).to(init_device)

            # Forward through GRU\Transformer:
            with torch.no_grad():
                zero_z2, _ = self.gru_layer(zero_z1_flat)

            # Needs to pool by the sequence dim; (batch, sequence, features) --> (batch, features, sequence):
            zero_z2_T = torch.transpose(zero_z2, 1, 2)
            # Define GRU\Transformer pooling:
            self.gru_pooling = nn.AvgPool1d(kernel_size=zero_z2_T.shape[2]//1).to(init_device)
            # Forward through pooling:
            with torch.no_grad():
                zero_z2_T_pooled = self.gru_pooling(zero_z2_T)
            # Flatten:
            zero_z2_flat = self.gru2pool(zero_z2_T_pooled, dummy_batch_size)

        # Define classifier:
        if self.encoder_only:
            classifier_in = zero_z1.shape[1]*zero_z1.shape[2]
        else:
            classifier_in = zero_z2_flat.shape[1]
        self.classifier = CompactClassifier(classifier_in, classifier_in, n_classes)

        # Since using GPU for the init forward pass, clear memory:
        torch.cuda.empty_cache()

    def input2windows(self, x):
        """
        Dimension modifications: signal input --> encoder input
        # split samples (length) to windows.
        # Transpose between leads and windows.
        # Concatenate batch into the window dim, allowing passing through 1D convolution.
        """
        batch_size, N_leads, N_samples = x.shape
        # split to windows:
        signal_windows = x.view(batch_size, N_leads, -1, int(self.fs * self.window_length * 60))  # 60 is seconds in minutes
        # (batch, lead, window, samples) --> (batch, window, lead, samples):
        signal_windows = torch.transpose(signal_windows, 1, 2)
        # (batch, window, lead, samples) --> (window, lead, samples) and concatenate batches in the window dim;
        signal_windows = signal_windows.reshape(-1, N_leads,  signal_windows.shape[3])
        return signal_windows

    def encoder2gru(self, z1, batch_size):
        """
        Dimension modifications: encoder output --> GRU input
        Split back to batch.
        Flatten channels and features.
        """
        # (window, channels, features) --> (batch, window, channels, features)
        z1_batch = z1.reshape(batch_size, -1, z1.shape[1], z1.shape[2])
        # Flatten features  --> (batch, sequence (window), features):
        z1_flat = z1_batch.view(batch_size,  z1_batch.shape[1], -1)
        return z1_flat

    @staticmethod
    def gru2pool(z2_T_pooled, batch_size):
        """
        Dimension modifications: GRU output --> pooling input
        Flatten.
        """
        z2_flat = z2_T_pooled.reshape(batch_size, z2_T_pooled.shape[1] * z2_T_pooled.shape[2])
        return z2_flat

    def forward(self, x, skip_classifier=False):
        batch_size, _, _ = x.shape
        signal_windows = self.input2windows(x)

        # Encoding:
        z1 = self.encoder(signal_windows)
        # Pooling:

        if not self.encoder_only:
            z1_flat = self.encoder2gru(z1, batch_size)
            # GRU\Transformer:
            z2, _ = self.gru_layer(z1_flat)
            z2_T = torch.transpose(z2, 1, 2)
            # Pooling:
            z2_T_pooled = self.gru_pooling(z2_T)
            z2_flat = self.gru2pool(z2_T_pooled, batch_size)

        if skip_classifier:
            return z2_flat
        else:
            # Linear
            if self.encoder_only:
                z1_flat = z1.view(z1.shape[0], -1)
                y = self.classifier(z1_flat)
            else:
                y = self.classifier(z2_flat)
            return y
