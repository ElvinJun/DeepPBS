import os
import shutil
import random
import pathlib


DATA_PATH = '/share/Data/processed/nr40/bitorsion'
SUBSET_PATH = '/share/Data/processed/nr40/10fold_val_subset'
subset_fold = 10


filenames = os.listdir(DATA_PATH)
random.shuffle(filenames)
print('Total file number = %d' % len(filenames))

for i in range(subset_fold):
    print('subset', i)
    train_path = os.path.join(SUBSET_PATH, 'subset_%d/train' % i)
    val_path = os.path.join(SUBSET_PATH, 'subset_%d/val' % i)
    pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(val_path).mkdir(parents=True, exist_ok=True)

    start_index = i / 10 * len(filenames) // 1
    end_index = (i + 1) / 10 * len(filenames) // 1
    print('from %d to %d' % (start_index, end_index))

    for k, filename in enumerate(filenames):
        if start_index <= k < end_index:
            shutil.copy(os.path.join(DATA_PATH, filename), os.path.join(val_path, filename))
        else:
            shutil.copy(os.path.join(DATA_PATH, filename), os.path.join(train_path, filename))
