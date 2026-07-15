import os
import random
import torch
import ast
import csv

class DummyDataset:
    def __init__(self, csv_file):
        self.labels = []
        self.subjects = []
        self.sets = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'label' in row:
                    labels = ast.literal_eval(str(row['label']))
                    self.labels.append(labels)
                    self.subjects.append(str(row['subject']))
                    self.sets.append(str(row.get('set', '')))
                
        self.labels = torch.tensor(self.labels).float()

    def __len__(self):
        return len(self.labels)

random.seed(42)
data_path = './data/deadlift_dataset.csv'
full_dataset = DummyDataset(data_path)

train_indices = []
valid_indices = []
test_indices = []

subject_to_sets = {}
for idx, (sub, set_val) in enumerate(zip(full_dataset.subjects, full_dataset.sets)):
    if sub not in subject_to_sets:
        subject_to_sets[sub] = {}
    if set_val not in subject_to_sets[sub]:
        subject_to_sets[sub][set_val] = []
    subject_to_sets[sub][set_val].append(idx)
    
remaining_sets = []

for sub, sets_dict in subject_to_sets.items():
    unique_sets = sorted(list(sets_dict.keys()))
    random.shuffle(unique_sets)
    
    train_set = unique_sets[0]
    train_indices.extend(sets_dict[train_set])
    
    for s in unique_sets[1:]:
        remaining_sets.append(sets_dict[s])
        
random.shuffle(remaining_sets)

total_data = len(full_dataset)
target_train = int(0.7 * total_data)
target_val = int(0.1 * total_data)
target_test = total_data - target_train - target_val

for indices in remaining_sets:
    def_train = target_train - len(train_indices)
    def_val = target_val - len(valid_indices)
    def_test = target_test - len(test_indices)
    
    max_def = max(def_train, def_val, def_test)
    
    if max_def == def_train:
        train_indices.extend(indices)
    elif max_def == def_val:
        valid_indices.extend(indices)
    else:
        test_indices.extend(indices)

train_labels = full_dataset.labels[train_indices]
val_labels = full_dataset.labels[valid_indices]
test_labels = full_dataset.labels[test_indices]

error_names = [
    'Barbell_moving_away_from_the_shins',
    'Hips_rising_before_the_barbell_leaves_the_ground',
    'Barbell_colliding_with_the_knees',
    'Lower_back_rounding'
]

print('\n📊 Set Split 訓練集 (Train) 標籤分佈:')
for i, name in enumerate(error_names):
    count = train_labels[:, i].sum().item()
    print(f' - {name}: {int(count)} 下')

print('\n📊 Set Split 驗證集 (Val) 標籤分佈:')
for i, name in enumerate(error_names):
    count = val_labels[:, i].sum().item()
    print(f' - {name}: {int(count)} 下')

print('\n📊 Set Split 測試集 (Test) 標籤分佈:')
for i, name in enumerate(error_names):
    count = test_labels[:, i].sum().item()
    print(f' - {name}: {int(count)} 下')

print(f'\n總資料量: Train={len(train_indices)}, Val={len(valid_indices)}, Test={len(test_indices)}')
