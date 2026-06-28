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
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            if 'features' in row and 'label' in row:
                features = ast.literal_eval(str(row['features']))
                labels = ast.literal_eval(str(row['label']))
                subject = str(row['subject'])
                self.features.append(torch.tensor(features).float())
                self.labels.append(torch.tensor(labels).float())
                self.subjects.append(subject)
        
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
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            if 'features' in row and 'label' in row:
                features = ast.literal_eval(str(row['features']))
                labels = ast.literal_eval(str(row['label']))
                subject = str(row['subject'])
                self.features.append(torch.tensor(features).float())
                self.labels.append(torch.tensor(labels).float())
                self.subjects.append(subject)
        
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


class Datasubset(Dataset):
    def __init__(self, dataset, indices, transform=False):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y, true_idx = self.dataset[self.indices[idx]]
        return x, y, true_idx
