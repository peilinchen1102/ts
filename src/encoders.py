from ECG_JEPA.models import load_encoder
import torch

ENCODER_DIRS = {
    'original': '../ECG_JEPA/multiblock_epoch100.pth',
    'Ga': '../ECG_JEPA/weights/ecg_jepa_20250114_183403_(0.175, 0.225)_Ga/epoch100.pth',
    'ChapmanShaoxing': '../ECG_JEPA/weights/ecg_jepa_20250114_173445_(0.175, 0.225)_Chapman/epoch100.pth',
    'all': '../ECG_JEPA/weights/ecg_jepa_20250114_214922_(0.175, 0.225)_all/epoch70.pth'
}

def get_encoder(model_name: str, device, encoder_name):
    if model_name == 'transformer':
        return None, None
    
    elif model_name == 'ecg-jepa':
        if encoder_name == 'random':
            encoder, embed_dim = load_encoder()
        else:
            encoder_dir = ENCODER_DIRS[encoder_name]
            encoder, embed_dim = load_encoder(encoder_dir)

        encoder.to(device)
        return encoder, embed_dim
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
