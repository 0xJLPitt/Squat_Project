import os
# Soft link the dataset
# ln -s /cats/datasets/BenchpressDataset ./BenchpressDataset
# ln -s /cats/datasets/DeadliftDataset_0408 ./DeadliftDataset
os.system("python PatchTST_train.py --sport benchpress --subject_split True --tag ")
os.system("python PatchTST_train.py --sport deadlift --subject_split True --tag ")
os.system("python PatchTST_train.py --sport benchpress --subject_split False --tag ")
os.system("python PatchTST_train.py --sport deadlift --subject_split False --tag ")
