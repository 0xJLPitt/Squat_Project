import torch.nn as nn
import torch
import os, glob, json
import numpy as np
from fitness_system_backend.settings import DEADLIFT_ERROR_MODEL_PATH

class PatchEmbedding(nn.Module):
    def __init__(self, input_len, patch_len, input_dim, embed_dim, stride=None):
        super().__init__()
        self.patch_len = patch_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.stride = stride if stride else patch_len
        self.proj = nn.Linear(patch_len * input_dim, embed_dim)

    def forward(self, x):
        B, T, F = x.shape
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x = x.contiguous().view(B, -1, self.patch_len * F)
        x = self.proj(x)
        return x

class PatchTSTClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, input_len, patch_len=10,
                    embed_dim=256, num_heads=4, num_layers=4, dropout=0.3, stride=1):
        super().__init__()
        self.patch_embed = PatchEmbedding(input_len, patch_len, input_dim, embed_dim, stride=stride)
        self.embed_dim = embed_dim

        num_patches = (input_len - patch_len) // (stride if stride else patch_len) + 1
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )


    def forward(self, x):
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # 動態調整位置嵌入的長度（安全做法）
        pos_embed = self.pos_embed[:, :x.size(1), :]  # 可裁切
        if pos_embed.size(1) < x.size(1):  # 若不足，補上隨機值（不常見）
            extra = torch.randn(1, x.size(1) - pos_embed.size(1), self.embed_dim, device=x.device)
            pos_embed = torch.cat([pos_embed, extra], dim=1)

        x = x + pos_embed  # (B, tokens, embed_dim)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)
    
def merge_data(folder):
    features = []
    delta_path = os.path.join(folder, 'filtered_delta_norm')
    delta2_path = os.path.join(folder, 'filtered_delta2_norm')
    square_path = os.path.join(folder, 'filtered_delta_square_norm')
    zscore_path = os.path.join(folder, 'filtered_zscore_norm')
    orin_path = os.path.join(folder, 'filtered_norm')

    if not all(
            map(os.path.exists,
                [delta_path, delta2_path, zscore_path, square_path, orin_path
                ])):
        print(f"Missing data in {folder}")
        return

    deltas = glob.glob(os.path.join(delta_path, '*.txt'))
    delta2s = glob.glob(os.path.join(delta2_path, '*.txt'))
    squares = glob.glob(os.path.join(square_path, '*.txt'))
    zscores = glob.glob(os.path.join(zscore_path, '*.txt'))
    orins = glob.glob(os.path.join(orin_path, '*.txt'))

    data_per_ind = list(fetch(zip(deltas, delta2s, zscores, squares,
                                orins)))  # Ensure list output
    features.extend(data_per_ind)
    return features


def fetch(uds):
    data_per_ind = []
    for ud in uds:
        parsed_data = []
        for file in ud:
            with open(file, 'r') as f:
                lines = f.read().strip().split('\n')
                parsed_data.append(
                    [list(map(float, line.split(','))) for line in lines])

        for num in zip(*parsed_data):
            data_per_ind.append([item for sublist in num for item in sublist])
            if len(data_per_ind) == 110:
                yield torch.tensor(data_per_ind,
                                    dtype=torch.float32)  # 確保是 Tensor
                data_per_ind = []


def predict(model, feature):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not isinstance(feature, torch.Tensor):
        feature = torch.as_tensor(feature, dtype=torch.float32)
    feature = feature.clone().detach().to(device)  # ✅ 修正方式
    feature = feature.unsqueeze(0)  # 增加 batch 維度
    with torch.no_grad():
        output = model(feature)  # 獲取模型輸出
        predicted_class = torch.argmax(output, dim=1)  # 取得最大信心值的類別
        confidence_scores = torch.softmax(output, dim=1)  # 計算分類信心值
    return predicted_class.item(), confidence_scores.cpu().numpy()


def save_to_config(y_data, output_file):
    # 遍歷數據並將 float32 轉為 Python 原生 float
    def convert(o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()  # 轉換為 Python list
        elif isinstance(o, np.float32):
            return float(o)  # 轉換為 Python float
        elif isinstance(o, torch.float32):
            return float(o)  # 轉換為 Python float
        return o

    config_data = {"results": y_data}  # 儲存 key

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, default=convert)

def run_predict(folder):
    output_file = os.path.join(folder, 'config', 'Score.json')

    category = {
        '2': 'Barbell_moving_away_from_the_shins',
        '3': 'Hips_rising_before_the_barbell_leaves_the_ground',
        '4': 'Barbell_colliding_with_the_knees',
        '5': 'Lower_back_rounding'
    }
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_path = os.path.join(folder, 'data_norm2')
    features = merge_data(data_path)
    for num, name in category.items():
        model = PatchTSTClassifier(input_dim=40, num_classes=4, input_len=110)
        state_dict = torch.load(
            DEADLIFT_ERROR_MODEL_PATH,
            map_location=device,
            weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        # 把每一下的結果丟入模型
        for i, feature in enumerate(features):
            if f"{i}" not in results:
                results[f"{i}"] = {}  # 初始化 key
            pred, conf = predict(model, feature)
            score = 1
            for idx, c in enumerate(conf[0]):
                results[f"{i}"][category[str(idx+2)]] = round(c, 4)
                score -= c * 0.25
            results[f"{i}"]["score"] = round(score, 4)
    save_to_config(results, output_file)
