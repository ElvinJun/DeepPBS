import os
import pathlib
import numpy as np
import shutil


train_name = 'nr40_Split_LeL2_Drop05'
cross_validation_fold = 10

top_models_folder = os.path.join('./outputs/%s_top_models_2' % train_name)
pathlib.Path(top_models_folder).mkdir(parents=True, exist_ok=True)

val_losses = []
for subset_index in range(cross_validation_fold):
    with open('./outputs/%s_%d/validation_map_2.txt' % (train_name, subset_index)) as file:
        lines = file.readlines()
    for line in lines[1::4]:
        val_losses.append(float(line.split('=')[1][:-1]))

total_epochs = len(val_losses) // cross_validation_fold
val_losses = np.array(val_losses).reshape(-1, total_epochs)

top_num = 5
for subset_index in range(cross_validation_fold):
    for top_index in np.argsort(val_losses[subset_index])[:top_num]:
        model_path = './outputs/%s_%d/%d_Linear.pth' % (train_name, subset_index, top_index)
        new_model_path = os.path.join(top_models_folder, 'model_%d.pth' % (top_index + subset_index * total_epochs))
        shutil.copy(model_path, new_model_path)
