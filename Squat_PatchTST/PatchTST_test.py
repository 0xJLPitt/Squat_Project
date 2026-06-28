import os
import torch
import time
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import json
import matplotlib.pyplot as plt
import numpy as np
from tools import *
import argparse
from torch.utils.data import DataLoader, random_split
from models import PatchTSTClassifier
from sklearn.metrics import accuracy_score


def test_model_with_path_tracking(model, test_loader, criterion, txt_dir, save_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()
    
    # **存放測試過程的數據**
    total_loss, total_time = 0.0, 0.0  
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels, indices in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.ndim != 3:
                raise ValueError(f"Expected 3D input (B, T, F), got shape {inputs.shape}")

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)  # [B, num_classes]
            preds = (probs > 0.5).int()     # [B, num_classes]
            
            y_true.extend(labels.cpu().numpy().tolist())    # labels shape: (batch, num_classes)
            y_pred.extend(preds.tolist())                   # preds shape: (batch, num_classes)

    avg_loss = total_loss / len(test_loader)
    avg_time_per_sample = total_time / len(y_true)
    f1 = f1_score(y_true, y_pred, average='macro')

    # 繪製混淆矩陣
    # classes = ['Correct', 'wrists bending backward', 'tilting to the right', 'tilting to the left', 'elbows flaring', 'scapular protraction']
    # classes = ['Correct', 'tilting to the right', 'tilting to the left', 'elbows flaring', 'scapular protraction']
    # binary_classes = ['The barbell is moving away from the shins.', 'Hips rise before the barbell leaves the ground.', 'The barbell collides with the knees.', 'Lower back rounding']
    classes = ['Correct', 'Far from the shins', 'Hips rise first', 'Collide with the knees', 'Lower back rounding']
    binary_classes = classes[1:]
    
    cm = multilabel_confusion_matrix(y_true, y_pred, sample_weight=None, labels=None, samplewise=False)
    n_classes = cm.shape[0]
    fig, axes = plt.subplots(nrows=(n_classes+1)//2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    for i in range(n_classes):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i], display_labels=['False', 'True'])
        disp.plot(include_values=True, cmap="Blues", ax=axes[i], 
                xticks_rotation="horizontal", values_format="d")
        axes[i].set_title(f'Class: {binary_classes[i]}')

    plt.tight_layout()
    plt.savefig(f"{txt_dir}/confusion_matrix.png")
    plt.close()
    
    cm = multilabel_confusion_matrix_mix(y_true, y_pred, num_classes)
    plot_custom_confusion_matrix(cm, classes, f"{txt_dir}/confusion_matrix_mix.png")
    accuracy = accuracy_score(y_true, y_pred)    

    return avg_loss, f1, avg_time_per_sample, accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str)
    args = parser.parse_args()
    
    from dataset import *
    if args.sport == 'deadlift':
        data_path = os.path.join(os.getcwd(), 'data', '3D_Real_Final')
        full_dataset = Dataset_TST_Deadlift(data_path)
        save_dir = f'./models/TST_Deadlift/12'
        num_classes = 4
        input_len = 110
    elif args.sport == 'benchpress':
        data_path = os.path.join(os.getcwd(), 'data', 'new_subject', 'data.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
            test_dataset  = Dataset_Benchpress(all_data)
        output_dir = './models/benchpress/TST_Benchpress/Exp1/no_wrist_press'
        save_dir = './models/benchpress/TST_Benchpress/9/no_wrist_press'
        num_classes = 4
        input_len = 100
    input_dim = test_dataset.dim
    print('Input dimention',input_dim)
    
    best_f1 = -1
    best_seed = None
    best_model_path = ""

    all_f1_scores = []
    cost_times = []
    accuracies = []
    seeds = [42, 2023, 7, 88, 100, 999]
    
    for se in seeds:
        set_seed(se)

        # 分割資料
        gen = torch.Generator().manual_seed(se)  # 為每個seed創建獨立生成器
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 訓練與測試
        model = PatchTSTClassifier(input_dim, num_classes, input_len).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()

        save_path = os.path.join(save_dir, f"PatchTST_model_seed{se}.pth")
        txt_dir = os.path.join(output_dir, f"seed{se}")
        os.makedirs(txt_dir, exist_ok=True)
        
        avg_loss, f1, avg_time_per_sample, accuracy = test_model_with_path_tracking(
            model, test_loader, criterion, txt_dir, save_path, num_classes
        )
        print(f"Seed {se} Test F1: {f1:.4f}, Accuracy: {accuracy:.4f}, cost {avg_time_per_sample} sec")
        all_f1_scores.append(f1)
        cost_times.append(avg_time_per_sample)
        accuracies.append(accuracy)

        if f1 > best_f1:
            best_f1 = f1
            best_seed = se
            best_model_path = save_path

    write_result(model, seeds, all_f1_scores, accuracies, cost_times, output_dir, best_f1, best_seed, best_model_path)