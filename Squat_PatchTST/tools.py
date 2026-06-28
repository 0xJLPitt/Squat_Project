import random
import numpy as np
import torch, os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多張 GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def compute_f1_score(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels, indices in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)  # [B, num_classes]
            preds = (probs > 0.5).int()     # [B, num_classes]
            y_true.extend(labels.cpu().numpy().tolist())    # labels shape: (batch, num_classes)
            y_pred.extend(preds.tolist())                   # preds shape: (batch, num_classes)

    return f1_score(y_true, y_pred, average='macro')

def multilabel_confusion_matrix_mix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=int)  # +1 for Category_0
    offset = 1  # shift actual class indices by +1
    dummy_class = 0  # Category_0 index

    for yt, yp in zip(y_true, y_pred):
        yt = np.array(yt)
        yp = np.array(yp)

        true_classes = np.where(yt == 1)[0]
        pred_classes = np.where(yp == 1)[0]

        if len(true_classes) == 0:
            true_classes = [dummy_class]
        else:
            true_classes = [c + offset for c in true_classes]

        if len(pred_classes) == 0:
            pred_classes = [dummy_class]
        else:
            pred_classes = [c + offset for c in pred_classes]

        for t in true_classes:
            for p in pred_classes:
                cm[t][p] += 1

    return cm

def plot_custom_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.set_title("Pastch TST Confusion Matrix")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, ha="right", rotation=45)
    ax.set_yticklabels(class_names)

    # 顯示數值與百分比
    cm_sum = cm.sum(axis=1, keepdims=True)  # 每一列總數
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            total = cm_sum[i][0]
            if total == 0:
                percentage = 0
            else:
                percentage = count / total * 100
            ax.text(j, i, f"{count}\n({percentage:.1f}%)",
                    ha="center", va="center",
                    color="white" if count > cm.max() * 0.5 else "black",
                    fontsize=10)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.colorbar(im, ax=ax)
    plt.savefig(save_path)
    plt.close()
    
def write_result(model, num_folds, all_f1_scores, accuracies, cost_times, save_dir, best_f1, best_seed, best_model_path):
    # 🔍 顯示結果 & 建立結果字串
    summary_lines = []
    summary_lines.append("\n✅ F1 scores from each Fold:")
    for se, f1, ac, co in zip(list(range(num_folds)), all_f1_scores, accuracies, cost_times):
        summary_lines.append(f"Seed {se}: F1 = {f1:.4f}, Accuracy: {ac:.4f}, cost time = {co:.6f} sec")

    summary_lines.append(f"\n📊 Average F1 Score: {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}")
    summary_lines.append(f"🏆 Best F1: {best_f1:.4f} from Fold {best_seed}")
    summary_lines.append(f"📁 Best model saved at: {best_model_path}")

    # 印出結果到 terminal
    for line in summary_lines:
        print(line)

    # 📄 寫入 txt 檔案
    txt_output_path = os.path.join(save_dir, "results_summary.txt")
    with open(txt_output_path, "w", encoding="utf-8") as f:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total parameters: {total}\n")
        f.write(f"Trainable parameters: {trainable}\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\n✅ 寫入完成：{txt_output_path}")