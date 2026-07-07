import torchvision.models as models
import torch.nn as nn
import torch
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)  # 單向 LSTM 輸出 hidden_dim
        
    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # 只取最後一層的 hidden state
        out = self.fc(hn[-1])  # 使用最後一層的 hidden state 作為輸入
        return out
    
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Bi-LSTM 輸出是 2 倍 hidden_dim
        
    def forward(self, x):
        _, (hn, _) = self.bilstm(x)
        out = self.fc(torch.cat((hn[-2], hn[-1]), dim=1))  # 拼接正向與反向 hidden state
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        identity = self.shortcut(x)  # 短路連接
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 殘差連接
        out = torch.relu(out)
        return out

class ResNet32(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(ResNet32, self).__init__()
        self.initial = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        # ResNet-32 主要包含 5 個 Block（共 32 層）
        self.layer1 = self.make_layer(64, 64, 5)
        self.layer2 = self.make_layer(64, 128, 5, downsample=True)
        self.layer3 = self.make_layer(128, 256, 5, downsample=True)
        self.layer4 = self.make_layer(256, 512, 5, downsample=True)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 壓縮時間維度
        self.fc = nn.Linear(512, num_classes)  # 最終分類

    def make_layer(self, in_channels, out_channels, num_blocks, downsample=False):
        layers = [ResidualBlock(in_channels, out_channels, downsample=downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.initial(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)  # [batch, channels, 1]
        x = x.squeeze(-1)  # 去掉最後的 1 維
        x = self.fc(x)  # 最終分類
        return x
    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, embed_dim, stride):
        super().__init__()
        # 在 Channel Independence 模式下，輸入維度永遠是 1
        self.proj = nn.Linear(patch_len * 1, embed_dim)
        self.stride = stride
        self.patch_len = patch_len

    def forward(self, x):
        # x shape: (B*C, T, 1)
        B_C, T, _ = x.shape
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride) 
        # x shape: (B*C, num_patches, 1, patch_len)
        x = x.reshape(B_C, -1, self.patch_len) # 展平 patch
        x = self.proj(x) # (B*C, num_patches, embed_dim)
        return x

class PatchTSTClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, input_len, patch_len=10, 
                 embed_dim=256, num_heads=4, num_layers=4, dropout=0.3, stride=1):
        super().__init__()
        
        # 修正點：這裡傳入 1，因為每個通道獨立處理
        self.patch_embed = PatchEmbedding(patch_len, embed_dim, stride)
        
        num_patches = (input_len - patch_len) // stride + 1
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim * num_patches * embed_dim), # 這裡要改成 51200
            nn.Linear(input_dim * num_patches * embed_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        
        # --- 論文核心：Channel Independence ---
        # 1. 重排維度: (B, T, C) -> (B, C, T) -> (B*C, T, 1)
        x = x.permute(0, 2, 1).reshape(B * C, T, 1)
        
        # 2. Patching & Embedding
        x = self.patch_embed(x)  # (B*C, num_patches, embed_dim)
        
        # 3. Transformer
        x = x + self.pos_embed
        x = self.transformer(x)
        
        # 4. 聚合資訊 (Readout)
        # 先做時間維度平均 (Global Average Pooling over patches)
        # x = x.mean(dim=1)  # (B*C, embed_dim)
        
        # 再做通道間的聚合: (B*C, embed_dim) -> (B, C, embed_dim) -> (B, embed_dim)
        # x = x.view(B, C, -1).mean(dim=1) 
        x = x.view(B, -1)
        # 5. 分類層
        return self.classifier(x)
    