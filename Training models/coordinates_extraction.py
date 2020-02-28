import os
import pathlib
import numpy as np


def extract_cif(path, filename):
    with open(os.path.join(path, filename), 'r') as file:
        message = file.readlines()
    coos = []
    for line in message:
        line = line.split()
        if line[3] != 'CB':
            x = line[10]
            y = line[11]
            z = line[12]
            coos.append([float(x), float(y), float(z)])
    coos = np.array(coos)
    return coos.astype('float32')


dataset_name = 'nr_40'
LIST_PATH = 'D:\protein_structure_prediction\data\dataset/nr_list/best_rebuild_nr40.txt'  # % dataset_name
DATA_PATH = 'D:\protein_structure_prediction\data\dataset/cif_remove_again'
COOR_PATH = 'D:\protein_structure_prediction\data\dataset/processed_data/%s/coordinates' % dataset_name

pathlib.Path(COOR_PATH).mkdir(parents=True, exist_ok=True)

with open(LIST_PATH, 'r') as file:
    filenames = file.read().split('\n')
finished_filenames = os.listdir(COOR_PATH)
finished_num = 0
for filename in finished_filenames:
    if filename in filenames:
        filenames.remove(filename)
        finished_num += 1
print('%d finished! %d to go!' % (finished_num, len(filenames)))


failed_filename = []
for filename in filenames:
    print(filename)

    coos = extract_cif(DATA_PATH, filename + '.cif')

    np.save(os.path.join(COOR_PATH, filename), coos)
