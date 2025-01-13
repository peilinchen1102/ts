import torch
import time
import json
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from src.datasets import get_dataset_transformer, get_dataset_jepa
from src.dataloaders import get_dataloader
from src.encoders import get_encoder
from src.models import get_model
# # ecg jepa
# from ECG_JEPA.linear_probe_utils import features_dataloader, LinearClassifier
# from ECG_JEPA.models import load_encoder
# from ECG_JEPA.ecg_data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(train_dataset, test_dataset, train_encoder, num_epochs, ckpt=None) -> None:

    encoder, embed_dim = get_encoder(model_name, device)
    train_loader, test_loader = get_dataloader(model_name, train_dataset, test_dataset, encoder)

    criterion = nn.CrossEntropyLoss()

    if train_encoder:
        assert encoder != None
        for param in encoder.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=0.001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model = get_model(model_name, embed_dim, n_labels, device)
    if ckpt:
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
    

    ########################################## TRAINING ###########################################

    print("Training started!")
    start_time = time.time()
    training_metrics = {"loss": []}

    for epoch in range(num_epochs):
        if train_encoder: encoder.train()
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if train_encoder:
                features = encoder.representation(inputs)
                outputs = model(features)
            else:
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        training_metrics["loss"].append(average_loss)
        print(f'Average Loss: {average_loss:.4f}')

        checkpoint_path = f'{save_dir}/epoch_{epoch + 1}.pth'

        if train_encoder:
            torch.save({
                "epoch": epoch + 1,
                "encoder": encoder.state_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss
            }, checkpoint_path)
        else:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": average_loss
            }, checkpoint_path)

        print(f'Model checkpoint saved at {checkpoint_path}')
        torch.cuda.empty_cache()


    ########################################## EVALUATION ###########################################
    print("Evaluation started!")

    encoder.eval()
    model.eval()

    y_true = []
    y_pred = []
    y_scores = []

    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
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

    metrics_path = f"{save_dir}/test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f)
    print(f"Test metrics saved to {metrics_path}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Time taken: {training_time} seconds")


if __name__ == "__main__":

    print("Pipeline started!")
    
    print("GPU:", torch.cuda.current_device())
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    seg_method = sys.argv[3]
    train_encoder = False
    fixed= True if seg_method == 'fix' else False

    save_dir = f"{dataset_name}_{model_name}_{seg_method}" + "_encoder_trained" if train_encoder else ''

    if model_name == 'transformer':
        train_dataset, test_dataset = get_dataset_transformer(dataset_name, fixed=fixed, window_size=10, overlap=5)
    elif model_name == 'jepa':
        train_dataset, test_dataset = get_dataset_jepa(dataset_name, reload=False)

    main(train_dataset, test_dataset, train_encoder=train_encoder, num_epochs=10)