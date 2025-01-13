from ECG_JEPA.models import load_encoder
import torch

def get_encoder(model_name: str, device):
    if model_name == 'transformer':
        return None, None
    
    elif model_name == 'ecg-jepa':
        encoder, embed_dim = load_encoder()
        if ckpt:
            encoder, embed_dim = load_encoder()
            ckpt_dir = '../ECG_JEPA/multiblock_epoch100.pth'
            ckpt = torch.load(ckpt_dir)
            encoder.load_state_dict(ckpt['encoder'])
        encoder.to(device)
        return encoder, embed_dim
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
