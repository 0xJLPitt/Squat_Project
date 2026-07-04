from torch.utils.data import Dataset
import torch
import random

class Dataset_Benchpress(Dataset):
    def __init__(self, csv_file):
        import pandas as pd
        import ast
        self.features = []
        self.labels = []
        self.subjects = []
        self.sets = []
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            if 'features' in row and 'label' in row:
                features = ast.literal_eval(str(row['features']))
                labels = ast.literal_eval(str(row['label']))
                subject = str(row['subject'])
                set_val = str(row.get('set', ''))
                self.features.append(torch.tensor(features).float())
                self.labels.append(torch.tensor(labels).float())
                self.subjects.append(subject)
                self.sets.append(set_val)
                print(subject)
        
        print('features_shape:', self.features.shape, type(self.features))
        print('labels_shape:', self.labels.shape, type(self.labels))
        
        self.features = torch.stack(self.features) if self.features else torch.tensor([])
        self.labels = torch.stack(self.labels) if self.labels else torch.tensor([])
        self.dim = self.features.shape[-1] if len(self.features) > 0 else 0
        print(self.dim)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y, idx

class Dataset_Deadlift(Dataset):
    def __init__(self, csv_file):
        import pandas as pd
        import ast
        self.features = []
        self.labels = []
        self.subjects = []
        self.sets = []
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            if 'features' in row and 'label' in row:
                features = ast.literal_eval(str(row['features']))
                labels = ast.literal_eval(str(row['label']))
                subject = str(row['subject'])
                set_val = str(row.get('set', ''))
                self.features.append(torch.tensor(features).float())
                self.labels.append(torch.tensor(labels).float())
                self.subjects.append(subject)
                self.sets.append(set_val)
        
        self.features = torch.stack(self.features) if self.features else torch.tensor([])
        self.labels = torch.stack(self.labels) if self.labels else torch.tensor([])
        self.dim = self.features.shape[-1] if len(self.features) > 0 else 0

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y, idx


class Datasubset(Dataset):
    def __init__(self, dataset, indices, transform=False):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y, true_idx = self.dataset[self.indices[idx]]
        
        # 實作 Data Augmentation
        if self.transform:
            # clone 避免改到原始 dataset 中的 tensor
            x = x.clone() 
            
            # 1. Jittering (100% 加入常態分佈雜訊)
            # 加大干擾強度，標準差改為 0.05
            noise = torch.randn_like(x) * 0.05
            x = x + noise
                
            # 2. Scaling (100% 特徵縮放)
            # 每個維度乘上 0.90 ~ 1.10 的隨機比例 (加大縮放範圍)
            scale = torch.empty(x.shape[-1]).uniform_(0.90, 1.10)
            x = x * scale
                
            # 將數值稍微 clip 避免超出合理的範圍太多
            x = torch.clamp(x, min=-1.5, max=1.5)
            
        return x, y, true_idx
