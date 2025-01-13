import torch
from src.transformer_model import TimeSeriesTransformer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.dataset import PTBXL
import matplotlib.pyplot as plt
from load_datasets.catalog import DATASET_DICT
import sys
import numpy as np
import time
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import json

# ecg jepa
from ECG_JEPA.linear_probe_utils import features_dataloader, LinearClassifier
from ECG_JEPA.models import load_encoder
from ECG_JEPA.ecg_data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(train_dataset, test_dataset) -> None:
    
    # TODO: preprocessing
    # print("train_dataset shape", train_dataset[0][1].shape)
    # seq_len, new_window_size, channels = train_dataset[0][1].shape
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=64, pin_memory=True)
    
    print("Train dataset ready")

    # input_dim = new_window_size * channels   # Number of features (12 channels + positional encoding) * window size
    # d_model = 64                             # Dimension of embeddings (output from transformer)
    # num_heads = 8                            # Number of attention heads
    # num_layers = 6                           # Number of encoder layers
    # dim_feedforward = 256                    # Feedforward layer dimension
    # dropout = 0.1                            # Dropout rate
    # num_classes = train_dataset.num_classes()
    # learning_rate = 1e-4
    num_epochs = 100

    # model = TimeSeriesTransformer(
    #     input_dim=input_dim,
    #     d_model=d_model,
    #     num_heads=num_heads,
    #     num_layers=num_layers,
    #     dim_feedforward=dim_feedforward,
    #     dropout=dropout,
    #     num_classes=num_classes,
    # ).to(device)
    
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # checkpoint = torch.load("ptbxl_cpd_window_epoch_10.pth")
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    ###################### ECG JEPA #########################
    encoder, embed_dim = load_encoder('../ECG_JEPA/multiblock_epoch100.pth')
    encoder.to(device)

    # allow encoder training
    for param in encoder.parameters():
        param.requires_grad = True
        
    # features extracted
    n_labels = len(np.unique(labels_train))
    # train_loader = features_dataloader(encoder, train_loader, batch_size=32, shuffle=True, device=device)
    model = LinearClassifier(embed_dim, n_labels).to(device)
        
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=0.001)

    ###################### ECG JEPA ###################   ######

    print("Training started!")

    start_time = time.time()
    training_metrics = {"loss": []}

    for epoch in range(num_epochs):
        encoder.train() # for training encoder
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            # idx, inputs, labels = batch
            inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            features = encoder.representation(inputs)
            outputs = model(features)

            # outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        training_metrics["loss"].append(average_loss)
        print(f'Average Loss: {average_loss:.4f}')

        checkpoint_path = f'{save_dir}_epoch_{epoch + 1}.pth'
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": average_loss
        }, checkpoint_path)
        print(f'Model checkpoint saved at {checkpoint_path}')

        torch.cuda.empty_cache()


    print("Training complete!")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")

    encoder.eval()
    model.eval()

    print("Prep test dataset")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=64, pin_memory=True)

    ################# ECG JEPA
    # test_loader = features_dataloader(encoder, test_loader, batch_size=32, shuffle=False, device=device)
    #############

    print("Evaluation starts!")
    y_true = []
    y_pred = []
    y_scores = []

    total = 0
    correct = 0

    with torch.no_grad():
        # for idx, inputs, labels in test_loader:
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            features = encoder.representation(inputs) # encoder tained
            outputs = model(features)
            # outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())  # Probabilities for all classes
        
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        torch.cuda.empty_cache()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test data: {accuracy:.2f}%')

    # Evaluation: Macro-averaged AUC, Sensitivity, and Specificity
    
    y_true_one_hot = np.zeros((len(y_true), y_scores[0].shape[0]))  # Convert to one-hot encoding
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1

    # AUC calculation for multilabel (macro-averaged)
    auc = roc_auc_score(y_true_one_hot, np.array(y_scores), average='macro')
    print(f'Macro-Averaged AUC: {auc:.4f}')

    # Sensitivity and Specificity per class
    num_classes = y_scores[0].shape[0]
    
    sensitivities = []
    specificities = []

    for i in range(num_classes):
        y_true_binary = y_true_one_hot[:, i]  # Binary ground truth for class i

        # Correct binary predictions based on the highest probability class
        y_pred_binary = (np.argmax(np.array(y_scores), axis=1) == i).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

        # Class-specific metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

        print(f'Class {i}: Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')

    # Macro-averaged metrics
    macro_sensitivity = np.mean(sensitivities)
    macro_specificity = np.mean(specificities)
    print(f'Macro-Averaged Sensitivity: {macro_sensitivity:.4f}')
    print(f'Macro-Averaged Specificity: {macro_specificity:.4f}')

    test_metrics = {
        "accuracy": accuracy,
        "macro_auc": auc,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "class_sensitivities": sensitivities,
        "class_specificities": specificities,
    }

    metrics_path = f"{save_dir}_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f)

    print(f"Test metrics saved to {metrics_path}")


if __name__ == "__main__":

    print("Pipeline started!")
    
    print("GPU:", torch.cuda.current_device())
    dataset_name = sys.argv[1]
    save_dir = sys.argv[2]
    dataset = DATASET_DICT[dataset_name]

    # train_dataset = dataset(base_root="../local_ecg", window_size=10, overlap=5, train=True, download=False, dataset_name=dataset_name)
    # test_dataset = dataset(base_root="../local_ecg", window_size=10, overlap=5, train=False, download=False, dataset_name=dataset_name)
    
    ########################## ECG JEPA ###################################
    config = {
        # 'data_dir': '../high_modality/ecg/physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018',
        'data_dir': '../high_modality/ecg/WFDB_PTBXL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/',
        'dataset': 'ptbxl',
        'task': 'multiclass'
    }
    data_dir = config['data_dir']
    dataset = config['dataset']
    task = config['task']
    waves_train, waves_test, labels_train, labels_test = waves_from_config(config,reduced_lead=True)
    train_dataset = ECGDataset(waves_train, labels_train)
    test_dataset = ECGDataset(waves_test, labels_test)
    ########################## ECG JEPA ###################################

    # train_dataset = dataset(base_root="../high_modality_local", window_size=10, overlap=5, train=True, download=False, dataset_name=dataset_name)
    # test_dataset = dataset(base_root="../high_modality_local", window_size=10, overlap=5, train=False, download=False, dataset_name=dataset_name)

    train_dataset = train_dataset
    test_dataset = test_dataset
    # print(f"Train dataset length: {train_dataset.subject_data.shape}")
    # print(f"Test dataset length: {test_dataset.subject_data.shape}")
    main(train_dataset, test_dataset)
