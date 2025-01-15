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
import os
import json
import argparse

def generate_model_name(model_structure, encoder, datasets):
    if len(datasets) == 4:
        dataset_str = 'all'
    else:
        dataset_str = "+".join(datasets)
    return f"{model_structure}/{encoder}_{dataset_str}"

def main(train_dataset, test_dataset, train_encoder, num_epochs, model_ckpt=None) -> None:

    encoder, embed_dim = get_encoder(base_model, device, encoder_name=encoder_name)
    train_loader, test_loader = get_dataloader(base_model, train_dataset, test_dataset, encoder, device, train_encoder)

    criterion = nn.CrossEntropyLoss()
    n_labels = 7
    model = get_model(base_model, embed_dim, n_labels, device)

    if train_encoder:
        assert encoder != None
        for param in encoder.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=0.001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    if model_ckpt:
        checkpoint = torch.load(model_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])

        if train_encoder:
            encoder.load_state_dict(checkpoint['encoder'])

    ########################################## TRAINING ###########################################
    start_time = time.time()

    if task != 'lineareval':
        print("Training started!")
        training_metrics = {"loss": []}

        for epoch in range(num_epochs):
            if train_encoder: 
                    encoder.train()
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
            
            if train_encoder:
                features = encoder.representation(inputs)
                outputs = model(features)
            else:
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

    name = '+'.join(dataset_names) if len(dataset_names) > 1 else dataset_names[0]
    metrics_path = f"{save_dir}/test_metrics_{name}.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f)
    print(f"Test metrics saved to {metrics_path}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Time taken: {training_time} seconds")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Pipeline started!")
    print("GPU:", torch.cuda.current_device())

    parser = argparse.ArgumentParser(description="ECG Pipeline")

    parser.add_argument('--task',
        default='lineareval',
        type=str,
        help='lineareval, end2end, finetune')

    parser.add_argument('--datasets',
        default= ['ptbxl', 'CPSC', 'ChapmanShaoxing', 'Ga'],
        nargs='+',
        help='array of dataset names')

    parser.add_argument('--model',
        default='ecg-jepa',
        type=str,
        help='model structure')

    parser.add_argument('--seg', 
        default='fix',
        type=str,
        help='segmentation method')

    parser.add_argument('--encoder',
        default='original',
        type=str,
        help='original, Ga, ChapmanShaoxing, CPSC, ptbxl, all')

    parser.add_argument('--model_ckpt',
        default='',
        type=str,
        help='model checkpoint')

    parser.add_argument('--epochs',
        default=10,
        type=int,
        help='num of epochs to train')

    args = parser.parse_args()

    task = args.task
    dataset_names = args.datasets
    base_model = args.model
    seg_method = args.seg
    encoder_name = args.encoder
    model_ckpt = args.model_ckpt
    num_epochs = args.epochs
    
    if task == 'lineareval':
        save_dir = os.path.dirname(model_ckpt)
    elif task == 'end2end':
        model_name = generate_model_name(base_model, encoder_name, dataset_names)
        save_dir = f"/home/peili/ts/models/end2end/{model_name}"
    else:
        model_name = generate_model_name(base_model, encoder_name, dataset_names)
        save_dir = f"/home/peili/ts/models/finetune/{model_name}"

    train_encoder = True if task == 'end2end' else False

    # if not ckpt:
    #     name = '+'.join(dataset_names)
    #     save_dir = f"{name}_{model_name}_{seg_method}" + ("_encoder_trained" if train_encoder else '')
    # elif eval_only:
    #     save_dir = ckpt.split('/')[0]
    # else:
    #     datasets_str = ckpt.split('_')[0]
    #     prev_datasets = set(datasets_str.split('+'))
    #     if len(dataset_names) == 1 and dataset_names[0] in prev_datasets:
    #          save_dir = f"{datasets_str}_{model_name}_{seg_method}" + ("_encoder_pretrain_Ga")
    #     else:
    #         datasets_str += ('+' + dataset_names[0] if not eval_only else '')
    #     save_dir = f"{datasets_str}_{model_name}_{seg_method}" + ("_encoder_trained" if train_encoder else '')


    os.makedirs(save_dir, exist_ok=True) 

    if base_model == 'transformer':
        train_dataset, test_dataset = get_dataset_transformer(dataset_names, fixed=fixed, window_size=10, overlap=5)
    elif base_model == 'ecg-jepa':
        train_dataset, test_dataset = get_dataset_jepa(dataset_names, reload=False)

    main(train_dataset, test_dataset, train_encoder=train_encoder, num_epochs=num_epochs, model_ckpt=model_ckpt)

    