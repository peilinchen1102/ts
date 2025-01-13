from torch.utils.data import Dataset, DataLoader
from ECG_JEPA.models import load_encoder
from ECG_JEPA.linear_probe_utils import features_dataloader, LinearClassifier

def get_dataloader(model_name: str, train_dataset, test_dataset, encoder, device):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=64, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=64, pin_memory=True)

    if model_name == 'transformer':
        return train_loader, test_loader

    train_loader = features_dataloader(encoder, train_loader, batch_size=32, shuffle=True, device=device)
    test_loader = features_dataloader(encoder, test_loader, batch_size=32, shuffle=False, device=device)
    
   