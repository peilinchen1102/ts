import torch
import torch.nn as nn
import torch.nn.functional as F
from ECG_JEPA.linear_probe_utils import features_dataloader, LinearClassifier
from ECG_JEPA.models import load_encoder

def get_model(model_name: str, input_dim: int, n_labels: int, device):
    if model_name == 'transformer':
        model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=64,
            num_heads=8,
            num_layers=6,
            dim_feedforward=256,
            dropout=0.1,
            num_classes=n_labels
        )

    elif model_name == 'ecg-jepa':
        model = LinearClassifier(input_dim, n_labels)

    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.to(device)
    return model


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        d_model: int,
        num_heads: int, 
        num_layers: int, 
        dim_feedforward: int, 
        dropout: int, 
        num_classes: int
    ):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_dim = input_dim  # 13 = 12 channels + 1 positional encoding
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_dim, self.d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )
        self.classification_layer = nn.Linear(d_model, num_classes)
        
        self.batch_first = True

    def forward(self, x):
        batch_size, seq_len, window_size, channels = x.shape
        flatten_x = x.view(batch_size, seq_len, window_size * channels)

        # window_size * channels
        x = self.input_embedding(flatten_x)
        output = self.transformer_encoder(x) 
        pooled_output = output.mean(dim=1)  # (batch_size, d_model)
        x = self.classification_layer(pooled_output) # [batch_size, num_classes]
        return x

