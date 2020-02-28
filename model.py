import math
import numpy as np
import os
from numpy import *
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pathlib
from pred_torsion import *
from rebulid import *
import shutil
import time
import random
import sys

# path = 'D:\\backbone_prediction'
# distance_window_path = os.path.join(path, 'distance_window')
# path_CA = sys.argv[1]
# logger.info(path_CA)
# path_CA = '/data/wwwroot/webserver/files/2019-11-19-07-20-19/CA_info'
path_CA = 'D:\\backbone_prediction\\CA_infoglp'
distance_window_path = path_CA.replace('CA_infoglp', 'distance_window_test1')
pathlib.Path(distance_window_path).mkdir(parents=True, exist_ok=True)
atoms_type = ['N', 'CA', 'C', 'O']

acid_normol = ['ALA', 'PHE', 'CYS', 'ASP', 'ASN',
                'GLU', 'GLN', 'GLY', 'HIS', 'LEU',
                'ILE', 'LYS', 'MET', 'PRO', 'ARG',
                'SER', 'THR', 'VAL', 'TRP', 'TYR']

ALPHABET = {'A': 'ALA', 'F': 'PHE', 'C': 'CYS', 'D': 'ASP', 'N': 'ASN',
            'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'L': 'LEU',
            'I': 'ILE', 'K': 'LYS', 'M': 'MET', 'P': 'PRO', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}

AA_HYDROPATHICITY_INDEX = {'ARG': -4.5, 'LYS': -3.9, 'ASN': -3.5, 'ASP': -3.5, 'GLN': -3.5,
                           'GLU': -3.5, 'HIS': -3.2, 'PRO': -1.6, 'TYR': -1.3, 'TRP': -0.9,
                           'SER': -0.8, 'THR': -0.7, 'GLY': -0.4, 'ALA': 1.8, 'MET': 1.9,
                           'CYS': 2.5, 'PHE': 2.8, 'LEU': 3.8, 'VAL': 4.2, 'ILE': 4.5}

AA_BULKINESS_INDEX = {'ARG': 14.28, 'LYS': 15.71, 'ASN': 12.82, 'ASP': 11.68, 'GLN': 14.45,
                      'GLU': 13.57, 'HIS': 13.69, 'PRO': 17.43, 'TYR': 18.03, 'TRP': 21.67,
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


# 提取CA原子信息
def atoms_infos(file_name):
    file = open(os.path.join(path_CA, file_name), 'r')
    lines = file.readlines()
    atoms_info = [line.strip('\n') for line in lines
                  if line.split()[0] == 'ATOM' and line.split()[2] == 'CA']
    array_head_tail = np.zeros((5, 3))

    for line in lines:
        if line.split()[0] == 'ATOM' and line.split()[2] == 'N':
            array_head_tail[0] = [float(line.split()[j]) for j in range(6, 9)]
            break

    for line in lines[::-1]:
        if line.split()[0] == 'ATOM' and line.split()[2] == 'C':
            array_head_tail[1] = [float(line.split()[j]) for j in range(6, 9)]
            break

    for line in lines[::-1]:
        if line.split()[0] == 'ATOM' and line.split()[2] == 'O':
            array_head_tail[2] = [float(line.split()[j]) for j in range(6, 9)]
            break

    for line in lines:
        if line.split()[0] == 'ATOM' and line.split()[2] == 'C':
            array_head_tail[3] = [float(line.split()[j]) for j in range(6, 9)]
            break

    for line in lines[::-1]:
        if line.split()[0] == 'ATOM' and line.split()[2] == 'N':
            array_head_tail[4] = [float(line.split()[j]) for j in range(6, 9)]
            break

    delet = []
    # 筛掉重复概率小的氨基酸
    for i in range(len(atoms_info)):
        if atoms_info[i - 1].split()[2] == atoms_info[i].split()[2] and \
                atoms_info[i - 1].split()[5] == atoms_info[i].split()[5]:
            if atoms_info[i - 1].split()[-3] <= atoms_info[i].split()[-3]:
                delet.append(i - 1)
            else:
                delet.append(i)
    for i in delet[::-1]:
        del atoms_info[i]
    return atoms_info,array_head_tail


# 提取坐标信息
def extract_coord(atoms_info):
    coord_array = np.zeros((len(atoms_info), 3))
    # coord_all = np.zeros((len(atoms_info), 3))
    acid_list = []
    CB_whether_exist = []
    for i in range(len(atoms_info)):

        #判断该氨基酸是否存在CB
        if atoms_info[i].split()[2] == 'CA':
            if atoms_info[i].split()[3] == 'GLY':
                CB_whether_exist.append('-')
            else:
                CB_whether_exist.append('+')
        coord_array[i] = [float(atoms_info[i].split()[j]) for j in range(6, 9)]
        acid_list.append(atoms_info[i].split()[3][-3::])

    acid_array = array(acid_list)
    return coord_array, acid_array, CB_whether_exist


def torsion():
    for n in range(len(torsion_sin)):
        torsion_training[n] = math.atan2(torsion_sin[n], torsion_cos[n])


def distance_window(coord_array, acid_array, i):
    WINDOW_SIZE = 15
    distCA = pdist(coord_array, metric='euclidean')
    distCA = squareform(distCA).astype('float32')
    # save_name = file_name.replace('pdb', 'npy')
    save_name = str(i) + '.npy'
    mark_type = [('distance', float), ('aa', 'S10')]
    dist_windows = []

    for i in range(len(distCA)):
        marked_array = []
        new_array = []
        for j in range(len(distCA[i])):
            marked_array.append((distCA[i, j], acid_array[j]))
        marked_array = np.array(marked_array, dtype=mark_type)
        marked_array = np.sort(marked_array, order='distance')[:WINDOW_SIZE]
        for j in range(len(marked_array)):
            aa = marked_array[j][1].decode('utf-8')
            new_array.append([marked_array[j][0]] + AA_MESSAGE[aa])
        dist_windows.append(new_array)
    dist_windows = np.array(dist_windows).astype('float32')

    np.save(os.path.join(distance_window_path, save_name), dist_windows)


#通过两坐标计算单位向量
def vector_unit(vector_1,vector_2):
    bond_vector_2 = vector_1 - vector_2
    bond_length_2 = np.linalg.norm(bond_vector_2)
    return bond_vector_2 / bond_length_2


def torsion_angle(A, B, C):
    #计算法向量
    U_2 = vector_unit(B,A); U_1 = vector_unit(C,B)#; U = vector_unit(D,C)
    # N   = np.cross(U_1, U)   / np.linalg.norm(np.cross(U_1, U))
    N_1 = np.cross(U_2, U_1) / np.linalg.norm(np.cross(U_2, U_1))
    m_weight = np.array([U_1, np.cross(N_1, U_1), N_1])

    #torsion_angle
    # try:
    #     angle = np.sign(np.dot(U_2,N)) * math.acos(np.dot(N_1,N))
    # except:
    #     angle = 0
    return m_weight


#根据真实角度或训练角度预测下一个坐标
def next_coord(A, B, C, R, angle_confirm,torsion_pred):
    #torsion_angle
    m = torsion_angle(A, B, C)
    #将真实角度或预测角度赋值给torsion
    torsion = torsion_pred
#     print("N——angle:",angle_real,angle_train)
    angle_martix=[math.cos(math.pi-angle_confirm),
                  math.sin(math.pi-angle_confirm) * math.cos(torsion),
                  math.sin(math.pi-angle_confirm) * math.sin(torsion)]
    #计算下一个坐标
    next_corrd = C + R * np.dot(m.T, angle_martix)

    return next_corrd


#计算预测的CB位置
def pred_CBcoord(N_coord, CA_coord, C_coord, CB_coord):
    # N和C的中间向量
    vector_midleline = (vector_unit(N_coord, CA_coord) + vector_unit(C_coord, CA_coord))
    vector_midleline_unit = vector_midleline / np.linalg.norm(vector_midleline)

    C = CA_coord + vector_midleline_unit * 0.841829775235248
    # C2 = N_coord + vector_unit(C_coord, N_coord) * 1.190426725853957
    angle_confirm = math.pi / 2

    # 统计CA到CB的距离: R = np.linalg.norm(C1 - CB_coord)
    R = 2.1545175870366853  # 统计得到的CA到CB的距离
    torsion = 0.5999114448494303  # 根据统计得到的旋转角

    # 计算得到预测的CB位置
    next_CB_coord = next_coord(CA_coord, N_coord, C, R, angle_confirm, torsion)
    return next_CB_coord


#将预测的CB加入数组
def add_pred_CB(pred_npy_without_CB, CB_whether_exist):
    pred_array = []
    for j in range(0, pred_npy_without_CB.shape[0]):
        # 进一步提取只有CB的array
        if j % 4 == 0:
            for line in range(4):
                pred_array.append(pred_npy_without_CB[j + line])
            if CB_whether_exist[j // 4] == '+':
                N_coord_pred = pred_npy_without_CB[j]
                CA_coord_pred = pred_npy_without_CB[j + 1]
                C_coord_pred = pred_npy_without_CB[j + 2]
                O_coord_pred = pred_npy_without_CB[j + 3]
                # 根据预测出的C和N计算得到CB的坐标
                next_CB = pred_CBcoord(N_coord_pred, CA_coord_pred, C_coord_pred, O_coord_pred)
                pred_array.append(next_CB)
    return array(pred_array)


def recovery_infos(pred_array, CA_infos, backbone_path):
    # pred_array = np.load(pred_array1)
    # CA_info = atoms_infos(pred_array1.split(".")[0]+".pdb")

    # after_work = open(pred_array1.split(".")[0] + "1" + ".pdb", "w")
    backbone = open(backbone_path, 'w')
# 完成pdb的框架
    list1 = []
    for i in range(len(CA_infos)):
        if CA_infos[i].split()[3] != "GLY":
            for j in range(5):
                list1.append(CA_infos[i])
        else:
            for j in range(4):
                list1.append(CA_infos[i])

    # 命名N\C\O\CB
    i = 0
    while i < len(list1) - 3:
        if list1[i].split()[3] == "GLY":
            list1[i] = list1[i].replace(list1[i].split()[2], "N ")
            list1[i + 2] = list1[i + 2].replace(list1[i + 2].split()[2], "C ")
            list1[i + 3] = list1[i + 3].replace(list1[i + 3].split()[2], "O ")
            i = i + 4

        else:
            list1[i] = list1[i].replace(list1[i].split()[2], "N ")
            list1[i + 2] = list1[i + 2].replace(list1[i + 2].split()[2], "C ")
            list1[i + 3] = list1[i + 3].replace(list1[i + 3].split()[2], "O ")
            list1[i + 4] = list1[i + 4].replace(list1[i + 4].split()[2], "CB")
            i = i + 5

    # 将npy的数据取三位小数
    for i in range(len(pred_array)):
        for j in range(3):
            pred_array[i][j] = "%.3f" % pred_array[i][j]

    # 坐标替换及补齐小数点位数
    for i in range(len(list1)):
        for j in range(3):
            if len(str(pred_array[i][j]).split(".")[1]) < 3:
                list1[i] = list1[i].replace(list1[i].split()[j + 6], str(pred_array[i][j]).split(".")[0] + "." + \
                                            str(pred_array[i][j]).split(".")[1].ljust(3, '0'))
            else:
                list1[i] = list1[i].replace(list1[i].split()[j + 6], str(pred_array[i][j]))

            # 最后一项原子名称修改
            list1[i] = list1[i].replace(list1[i].split()[11], list1[i].split()[2][0])

        # 序号与格式
        t = list1[i].split()
        list1[i] = t[0].ljust(7, ' ') + str(i + 1).rjust(4, ' ') + "  " + t[2].ljust(3, ' ') + t[3].rjust(4,
                                                                                                          ' ') + " " + \
                   t[4].ljust(2, ' ') + t[5].rjust(3, ' ') + t[6].rjust(12, ' ') + t[7].rjust(8, ' ') + t[8].rjust(8,
                                                                                                                   ' ') + \
                   "  " + t[9].ljust(5, ' ') + t[10].ljust(16, ' ') + t[11]

    for e in list1:
        backbone.write(e + "\n")
    backbone.close()

if __name__ == "__main__":
    CB_whether_exist_all = []
    #提取坐标信息计算windows_distance
    # time_statics = np.zeros((100, 4))
    count = 0
    # book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    # sheet = book.add_sheet('time_statics', cell_overwrite_ok=True)
    # for iter in range(100):
    start1 = time.time()
    # for file_name in os.listdir(path_CA):
    f = open('D:\\backbone_prediction\\CA_infoglp\\orign.txt', 'r')
    file = f.readlines()
    seqs = [seq.split()[0] for seq in file]
    # filename = [seq.split()[0] for seq in file]
    for i, seq in enumerate(seqs):
        try:
            # file_name = filename[i] + '.pdb'
            file_name = '4avz.pdb'
            atoms_info, ground_true_coos = atoms_infos(file_name)
            coord_array, acid_array, CB_whether_exist = extract_coord(atoms_info)
            # CB_whether_exist_all.append(CB_whether_exist)

            distance_window(coord_array[:223], seq, str(i))
            test_dataset = DistanceWindow(
                distance_window_path=distance_window_path)
            data_loader = DataLoader(dataset=test_dataset)
        except Exception as e:
            print(e)

    end1 = time.time()
    # 融合50个模型的角度

    models_path = 'D:\\backbone_prediction\\top_models'

    total_acid = 0

    with torch.no_grad():

        models = []
        start2 = time.time()
        for model_name in os.listdir(models_path):
            model = torch.load(os.path.join(models_path, model_name), map_location='cuda:0')
            model.eval()
            model.is_training = False
            models.append(model)
        end2 = time.time()

        start3 = time.time()
        for arrays, torsions, output_filename in data_loader:
            total_file = 0

            for model in models:
                arrays = arrays.to(device)
                pred_sincos = model(arrays[0]).squeeze(1).transpose(0, 1)
                output = pred_sincos.data.cpu().numpy()
                total_file += output
            total_file = total_file / 50

            # 根据预测角度复原坐标
            start4 = time.time()
            filename = output_filename[0]
            coos = []

            # 读入CA数据
            atoms_info, ground_true_coos_real = atoms_infos(filename.replace('npy', 'pdb'))
            coord_array, acid_array, CB_whether_exist = extract_coord(atoms_info)
            for coo in coord_array:
                coos.append(Coordinate(coo))

            PATH_OUTPUT = path_CA.replace('CA_info', 'backbone')
            pathlib.Path(PATH_OUTPUT).mkdir(parents=True, exist_ok=True)

            pred = total_file
            torsions_C = np.arctan2(pred[0], pred[1])
            torsions_N = np.arctan2(pred[2], pred[3])
            # 复原骨架结构
            backbone_pred_without_CB = backbone_rebuild_separated_torsion(coos, torsions_C, torsions_N)

            ground_true_coos = np.zeros((3, 3))
            # print(backbone_pred_without_CB[1],backbone_pred_without_CB[-2])
            if (ground_true_coos[0] == np.zeros((1, 3))).all():
                ground_true_coos[0] = next_coord(coord_array[1], backbone_pred_without_CB[1], coord_array[0],
                                                             1.45801, 2.124, 2.7)

            if (ground_true_coos[1] == np.zeros((1, 3))).all():
                ground_true_coos[1] = next_coord(coord_array[-2], backbone_pred_without_CB[-2], coord_array[-1],
                                                             1.52326, 1.941, -1.4)

            if (ground_true_coos[2] == np.zeros((1, 3))).all():
                ground_true_coos[2] = next_coord(coord_array[-2], backbone_pred_without_CB[-2], coord_array[-1],
                                                             2.408748478225743, 1.4915450962173677, -1.4)

            # loss[iter][count] = np.array([np.linalg.norm(ground_true_coos[i] - ground_true_coos_real[i]) for i in range(3)])

            pred_npy_without_CB = np.concatenate((ground_true_coos[0].reshape([1, 3]),
                                                  backbone_pred_without_CB, ground_true_coos[-2:]), axis=0).astype(
                'float32')

            CB_whether_exist = CB_whether_exist_all[count]
            count += 1
            pred_array = add_pred_CB(pred_npy_without_CB, CB_whether_exist)

            backbone_path = os.path.join(PATH_OUTPUT, filename.replace('npy', 'pdb'))
            recovery_infos(pred_array, atoms_info, backbone_path)
            end4 = time.time()
            time_total = end4 - start4
        end3 = time.time()
        # time_statics[iter] = np.array([end1-start1, end2-start2, end3-start3-time_total, time_total])
        print(end1-start1, end2-start2, end3-start3-time_total, time_total)

    # 将loss写入excel
    # loss_mean = np.mean(loss, axis=0)
    # for i in range(loss_mean.shape[0]):
    #     for j in range(loss_mean.shape[1]):
    #         sheet.write(i+1,j+1,loss_mean[i][j])
    # book.save('D://backbone_prediction//NC_random1.xls') n
    # print(time_statics)
    # for i in range(100):
    #     sheet.write(i + 1, 0, i+1)
    #     for j in range(4):
    #         sheet.write(i+1, j+1, time_statics[i][j])
    # book.save('D://backbone_prediction//time_stattics_cpu.xls')
