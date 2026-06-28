from random import choice
import torch
import os, json
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import PatchTSTClassifier
from sklearn.metrics import f1_score
from tools import *
import argparse
from PatchTST_test import test_model_with_path_tracking
import math

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path, num_epochs=150, patience=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_f1 = 0.0  # 用來儲存最佳 F1-score
    patience_counter = 0

    # **存放訓練過程的數據**
    train_losses = []
    train_f1_scores = []
    val_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        y_true, y_pred = [], []

        for inputs, labels, indices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)  # [B, num_classes]
            preds = (probs > 0.5).int()     # [B, num_classes]
            y_true.extend(labels.cpu().numpy().tolist())    # labels shape: (batch, num_classes)
            y_pred.extend(preds.tolist())                   # preds shape: (batch, num_classes)

        avg_loss = total_loss / len(train_loader)
        train_f1 = f1_score(y_true, y_pred, average='macro')
        val_f1 = compute_f1_score(model, valid_loader)

        scheduler.step()
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # **紀錄數據**
        train_losses.append(avg_loss)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        # 根據 F1-score 儲存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print("✅ Model Saved (Best F1-score)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early Stopping Triggered")
                break

    # **繪製 Loss 和 F1-score**
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label="Train Loss", color='blue', marker='o')
    plt.plot(epochs, train_f1_scores, label="Train F1-score", color='green', marker='s')
    plt.plot(epochs, val_f1_scores, label="Validation F1-score", color='red', marker='d')

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training Loss & F1-score per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")  # 儲存高解析度圖片

def get_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    max_epochs: int,
    min_lr_ratio: float = 0.0,
):
    """
    warmup_epochs:   線性 warmup epoch 數（從 0 -> base_lr）
    max_epochs:      總訓練 epoch 數
    min_lr_ratio:    最小 lr / base_lr，比方 0.0 就是衰到 0
    """
    assert warmup_epochs < max_epochs

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # 線性 warmup: 0 -> 1
            return float(current_epoch + 1) / float(warmup_epochs)
        # cosine decay: 1 -> min_lr_ratio
        progress = (current_epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str, choices=['benchpress', 'deadlift'])
    parser.add_argument('--subject_split', type=bool, help='Whether to split the dataset by subject')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of subset workers for DataLoader')
    parser.add_argument('--tag', type=str, help='Tag for save_dir, default is your data argumentation') # spawner, ...
    args = parser.parse_args()
    seeds = [42, 2023, 7, 88, 100, 999]
    
    from dataset import *
    
    if args.sport == 'deadlift':
        data_path = os.path.join(os.getcwd(), 'data', 'deadlift_dataset.csv')
        full_dataset = Dataset_Deadlift(data_path)
        save_dir = f'./models/deadlift/TST_Deadlift/{args.tag}'
        num_classes = 4
        input_len = 110
        
    elif args.sport == 'benchpress':
        data_path = os.path.join(os.getcwd(), 'data', 'benchpress_dataset.csv')
        full_dataset = Dataset_Benchpress(data_path)
        save_dir = f'./models/benchpress/TST_Benchpress/{args.tag}'
        num_classes = 4
        input_len = 100
        
    dataset_folds = []
    if args.subject_split:
        unique_subjects = sorted(list(set(full_dataset.subjects)))
        random.shuffle(unique_subjects)

        # 7:1:2 split for subjects
        n_subs = len(unique_subjects)
        tr_end = int(0.7 * n_subs)
        vl_end = int(0.8 * n_subs)
        
        train_subs = set(unique_subjects[:tr_end])
        val_subs = set(unique_subjects[tr_end:vl_end])
        test_subs = set(unique_subjects[vl_end:])
        
        train_indices = [idx for idx, s in enumerate(full_dataset.subjects) if s in train_subs]
        valid_indices = [idx for idx, s in enumerate(full_dataset.subjects) if s in val_subs]
        test_indices = [idx for idx, s in enumerate(full_dataset.subjects) if s in test_subs]
        
        # We only need one "fold" for a fixed 7:2:1 split
        dataset_folds = [(train_indices, valid_indices, test_indices)]
        num_folds = 1
    else:
        num_folds = len(seeds)
        for se in seeds:
            random.seed(se)
            all_indices = list(range(len(full_dataset)))
            random.shuffle(all_indices)
            
            n_total = len(all_indices)
            tr_end = int(0.7 * n_total)
            vl_end = int(0.8 * n_total)
            
            train_idx = all_indices[:tr_end]
            val_idx = all_indices[tr_end:vl_end]
            test_idx = all_indices[vl_end:]
            dataset_folds.append((train_idx, val_idx, test_idx))
            
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    cost_times = []
    accuracies = []

    for i, (t_idx, v_idx, test_indices) in enumerate(dataset_folds):
        train_dataset = Datasubset(full_dataset, t_idx, transform=True)
        valid_dataset = Datasubset(full_dataset, v_idx, transform=False)
        test_dataset = Datasubset(full_dataset, test_indices, transform=False)

        input_dim = full_dataset.dim
        print(f'Fold {i} | Input Dim: {input_dim} | Train: {len(train_dataset)}, Val: {len(valid_dataset)}, Test: {len(test_dataset)}')

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # 訓練與測試
        model = PatchTSTClassifier(input_dim, num_classes, input_len).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0003)
        criterion = torch.nn.BCEWithLogitsLoss()
        scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=5, max_epochs=100, min_lr_ratio=0.0)

        save_path = os.path.join(save_dir, f"PatchTST_model_fold{i}.pth")
        txt_dir = os.path.join(save_dir, f"PatchTST_model_fold{i}_results")
        fig_path = os.path.join(txt_dir, f"train_results_fold{i}.png")
        os.makedirs(txt_dir, exist_ok=True)

        train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, save_path, fig_path)

        avg_loss, f1, avg_time_per_sample, accuracy = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, num_classes
        )
        print(f"Fold {i} Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, cost {avg_time_per_sample} sec")
        all_f1_scores.append(f1)
        cost_times.append(avg_time_per_sample)
        accuracies.append(accuracy)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = i
            best_model_path = save_path

    write_result(model, num_folds, all_f1_scores, accuracies, cost_times, save_dir, best_f1, best_seed, best_model_path)