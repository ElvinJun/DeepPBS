import os
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pathlib


ALPHABET = {'A': 'ALA', 'F': 'PHE', 'C': 'CYS', 'D': 'ASP', 'N': 'ASN',
            'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'L': 'LEU',
            'I': 'ILE', 'K': 'LYS', 'M': 'MET', 'P': 'PRO', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
AA_HYDROPATHICITY_INDEX = {'ARG': -4.5, 'LYS': -3.9, 'ASN': -3.5, 'ASP': -3.5, 'GLN': -3.5,
                           'GLU': -3.5, 'HIS': -3.2, 'PRO': -1.6, 'TYR': -1.3, 'TRP': -0.9,
                           'SER': -0.8, 'THR': -0.7, 'GLY': -0.4, 'ALA': 1.8, 'MET': 1.9,
                           'CYS': 2.5, 'PHE': 2.8, 'LEU': 3.8, 'VAL': 4.2, 'ILE': 4.5}
AA_BULKINESS_INDEX = {'ARG': 14.28, 'LYS': 15.71, 'ASN': 12.82, 'ASP': 11.68, 'GLN': 14.45,
                      'GLU': 13.57, 'HIS': 13.69,  'PRO': 17.43, 'TYR': 18.03, 'TRP': 21.67,
                      'SER': 9.47, 'THR': 15.77, 'GLY': 3.4, 'ALA': 11.5, 'MET': 16.25,
                      'CYS': 13.46, 'PHE': 19.8, 'LEU': 21.4, 'VAL': 21.57, 'ILE': 21.4}
AA_FLEXIBILITY_INDEX = {'ARG': 2.6, 'LYS': 1.9, 'ASN': 14., 'ASP': 12., 'GLN': 4.8,
                        'GLU': 5.4, 'HIS': 4., 'PRO': 0.05, 'TYR': 0.05, 'TRP': 0.05,
                        'SER': 19., 'THR': 9.3, 'GLY': 23., 'ALA': 14., 'MET': 0.05,
                        'CYS': 0.05, 'PHE': 7.5, 'LEU': 5.1, 'VAL': 2.6, 'ILE': 1.6}
AA_MESSAGE = {}
for aa_short in ALPHABET.keys():
    aa_long = ALPHABET[aa_short]
    AA_MESSAGE.update({aa_short: [(5.5 - AA_HYDROPATHICITY_INDEX[aa_long]) / 10,
                                  AA_BULKINESS_INDEX[aa_long] / 21.67,
                                  (25. - AA_FLEXIBILITY_INDEX[aa_long]) / 25.]})
    AA_MESSAGE.update({aa_long: [(5.5 - AA_HYDROPATHICITY_INDEX[aa_long]) / 10,
                                 AA_BULKINESS_INDEX[aa_long] / 21.67,
                                 (25. - AA_FLEXIBILITY_INDEX[aa_long]) / 25.]})


def extract_pn(path, filename):
    with open(os.path.join(path, filename), 'r') as file:
        message = file.readlines()
    ca_coos = []
    seq_array = []
    seq = message[3][:-1]
    x = message[27][:-1].split('\t')
    y = message[28][:-1].split('\t')
    z = message[29][:-1].split('\t')
    mask = message[31][:-1]
    for i in range(len(mask)):
        if mask[i] == '+':
            ca_coos.append([float(x[3 * i + 1]) / 100., float(y[3 * i + 1]) / 100., float(z[3 * i + 1]) / 100.])
            aa = seq[i]
            seq_array.append(aa)
    ca_coos = np.array(ca_coos)
    seq_array = np.array(seq_array)
    return ca_coos, seq_array


def extract_cif(path, filename):
    with open(os.path.join(path, filename), 'r') as file:
        message = file.readlines()
    ca_coos = []
    seq_array = []
    # for line in message[1::3]:
    for line in message:
        line = line.split()
        if line[3] == 'CA':
            x = line[10]
            y = line[11]
            z = line[12]
            ca_coos.append([float(x), float(y) , float(z)])
            aa = line[5]
            seq_array.append(aa)
    ca_coos = np.array(ca_coos)
    seq_array = np.array(seq_array)
    return ca_coos, seq_array


dataset_name = 'test_set'

DATA_PATH = 'D:\protein_structure_prediction\data\dataset/test_set_atom_text'
DISTANCE_MAP_PATH = 'D:\protein_structure_prediction\data\dataset/processed_data/%s/distance_map' % dataset_name
DISTANCE_WINDOW_PATH = 'D:\protein_structure_prediction\data\dataset/processed_data/%s/distance_window' % dataset_name

pathlib.Path(DISTANCE_MAP_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(DISTANCE_WINDOW_PATH).mkdir(parents=True, exist_ok=True)


failed_filename = []

for filename in ['4FBR.npy']:
    filename = filename.replace('.npy', '.cif')
    print(filename)

    ca_coo_test, seq_test = extract_cif(DATA_PATH, filename)
    
def distance_window(coord_array):
    WINDOW_SIZE = 15
    distCA = pdist(ca_coo_test, metric='euclidean')
    distCA = squareform(distCA).astype('float32')

    save_name = filename.replace('.cif', '.npy')
    np.save(os.path.join(DISTANCE_MAP_PATH, save_name), distCA)

    mark_type = [('distance', float), ('aa', 'S10')]
    dist_windows = []
    for i in range(len(distCA)):
        marked_array = []
        new_array = []
        for j in range(len(distCA[i])):
            marked_array.append((distCA[i, j], seq_test[j]))
        marked_array = np.array(marked_array, dtype=mark_type)
        marked_array = np.sort(marked_array, order='distance')[:WINDOW_SIZE]
        for j in range(len(marked_array)):
            aa = marked_array[j][1].decode('utf-8')
            new_array.append([marked_array[j][0]] + AA_MESSAGE[aa])
        dist_windows.append(new_array)
    dist_windows = np.array(dist_windows).astype('float32')
    
    np.save(os.path.join(DISTANCE_WINDOW_PATH, save_name), dist_windows)
    




