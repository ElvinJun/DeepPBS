import numpy as np
import math
import os
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


L1_C = 0.5511235634596036
L1_N = 0.5275157666844377
# trans
L2_C_trans = 1.4281242923706199
R_C_trans = 0.5298886988235514
L2_N_trans = 1.4076846053568244
R_N_trans = 0.3797594132360668
L_O_trans = 1.6340680296346668
R_O_trans = 1.7458955685095028
# cis
L2_C_cis = 0.7914339670632375
R_C_cis = 1.309401495255961
L2_N_cis = 0.7973937679940248
R_N_cis = 1.2349344835918588
L_O_cis = 0.17424337647795887
R_O_cis = 2.384890116717385


class Coordinate(object):
    def __init__(self, coo):
        self.coo = coo
        self.x = self.coo[0]
        self.y = self.coo[1]
        self.z = self.coo[2]
        self.len = np.linalg.norm(self.coo)
        if self.len != 0:
            self.orient = self.coo/self.len


def vec(a, b):
    ab = b.coo - a.coo
    return Coordinate(ab)


def get_coo(line):
    items = line.split()
    x = float(items[10])
    y = float(items[11])
    z = float(items[12])
    return Coordinate(np.array([x, y, z]))


def get_coos(lines):
    atom_coos = []
    for line in lines:
        if line.split()[3] != 'CB':
            atom_coos.append(get_coo(line))
            # print(line.split()[3], len(atom_coos) % 4)
    return atom_coos


def read_pn(lines):
    x = lines[27].split('\t')
    y = lines[28].split('\t')
    z = lines[29].split('\t')
    mask = lines[31]
    atoms_coo = []
    for i in range(len(mask) * 3):
        if mask[i // 3] == '+':
            atoms_coo.append(Coordinate(np.array([float(x[i]) / 100., float(y[i]) / 100., float(z[i]) / 100.])))
            if atoms_coo[-1].len == 0:
                return None
    return atoms_coo


def get_cos(cb, cd):
    cos = np.dot(cb.coo, cd.coo)/(cb.len * cd.len)
    return cos


def get_angle(cb, cd):
    angle = math.acos(get_cos(cb, cd))
    return angle


def angle_norm(angle):
    normed_angle = math.atan2(math.sin(angle), math.cos(angle))
    return normed_angle


def array_angle_norm(array):
    normed_array = []
    for angle in array:
        normed_array.append(angle_norm(angle))
    return np.array(normed_array)


def get_projection(vec, axis):
    projection = Coordinate(vec.len * get_cos(vec, axis) * axis.orient)
    return projection


def get_sign(vec, axis):
    sign = Coordinate(vec.coo - get_projection(vec, axis).coo)
    return sign


# 计算以axis为轴，向量A到向量B的旋转角
def torsion(vector_A, vector_B, axis):
    N = Coordinate(np.cross(axis, vector_B))
    N_1 = Coordinate(np.cross(vector_A, axis))
    torsion = np.sign(np.dot(vector_A, N.orient)) * math.acos(np.dot(N_1.orient, N.orient))
    return torsion


# 计算夹角和坐标转换权重
def torsion_m(vector_A, axis):
    #计算法向量
    N_1 = Coordinate(np.cross(vector_A,axis)).orient
    #旋转基向量
    m_weight = np.array([axis , np.cross(N_1,axis) , N_1])
    angle = math.acos(np.dot(axis,vector_A))
    return m_weight, angle


# 根据向量，旋转轴 旋转角 计算旋转过后的向量
def rotation(vector_A, axis, torsion):
    m, angle = torsion_m(vector_A, axis)
    rotation_martix=[math.cos(math.pi-angle),
                  math.sin(math.pi-angle) * math.cos(torsion),
                  math.sin(math.pi-angle) * math.sin(torsion)]

    #计算旋转后向量
    vector_B = np.dot(m.T, rotation_martix)
    return vector_B


def distance_martix(A):
    # A是一个向量矩阵：euclidean代表欧式距离
    distA=pdist(A, metric='euclidean')
    # 将distA数组变成一个矩阵
    distB = squareform(distA)
    return distB


def backbone_rebuild_separated_torsion(coos, torsions_C, torsions_N):
    # coos: coordinates of CA only
    output_coos = [coos[0].coo]

    for k in range(len(coos) - 2):
        CA1 = coos[k]
        CA2 = coos[k + 1]
        CA3 = coos[k + 2]
        CA2CA3 = vec(CA2, CA3)
        CA1CA2 = vec(CA1, CA2)

        initial_orient = get_sign(CA2CA3, CA1CA2).orient
        axis = CA1CA2.orient
        torsion_pred_C = torsions_C[k]
        torsion_pred_N = torsions_N[k]
        if CA1CA2.len > 3.4:
            L2_C, L2_N, L_O, R_C, R_N, R_O = L2_C_trans, L2_N_trans, L_O_trans, R_C_trans, R_N_trans, R_O_trans
            torsion_pred_N = angle_norm(torsion_pred_N - math.pi)
        else:
            L2_C, L2_N, L_O, R_C, R_N, R_O = L2_C_cis, L2_N_cis, L_O_cis, R_C_cis, R_N_cis, R_O_cis

        output_C1 = CA1.coo + L2_C * CA1CA2.orient + R_C * rotation(initial_orient, axis, torsion_pred_C)
        output_O1 = CA1.coo + L_O * CA1CA2.orient + R_O * rotation(initial_orient, axis, torsion_pred_C)
        output_N2 = CA2.coo - L2_N * CA1CA2.orient + R_N * rotation(initial_orient, axis, torsion_pred_N)
        output_coos += [output_C1, output_O1, output_N2, CA2.coo]

    CA1 = coos[-3]
    CA2 = coos[-2]
    CA3 = coos[-1]
    CA2CA3 = vec(CA2, CA3)
    CA2CA1 = vec(CA2, CA1)

    initial_orient = get_sign(CA2CA1, CA2CA3).orient
    axis = CA2CA3.orient
    torsion_pred_C = torsions_C[-1]
    torsion_pred_N = torsions_N[-1]
    if CA2CA3.len > 3.4:
        L2_C, L2_N, L_O, R_C, R_N, R_O = L2_C_trans, L2_N_trans, L_O_trans, R_C_trans, R_N_trans, R_O_trans
        torsion_pred_N = angle_norm(torsion_pred_N - math.pi)
    else:
        L2_C, L2_N, L_O, R_C, R_N, R_O = L2_C_cis, L2_N_cis, L_O_cis, R_C_cis, R_N_cis, R_O_cis

    output_C2 = CA2.coo + L2_C * CA2CA3.orient + R_C * rotation(initial_orient, axis, torsion_pred_C)
    output_O2 = CA2.coo + L_O * CA2CA3.orient + R_O * rotation(initial_orient, axis, torsion_pred_C)
    output_N3 = CA3.coo - L2_N * CA2CA3.orient + R_N * rotation(initial_orient, axis, torsion_pred_N)
    output_coos += [output_C2, output_O2, output_N3, CA3.coo]
    output_coos = np.array(output_coos)
    return output_coos


# PATH = 'D:\protein_structure_prediction\data\dataset/test_set_withO'
PATH = 'D:\protein_structure_prediction\data\dataset/processed_data/test_set\coordinates'
REBUILD_PATH = 'D:\protein_structure_prediction\data\dataset/processed_data/test_set/rebuild_coordinates'
BITORSION_PATH = 'D:\protein_structure_prediction\data\dataset/processed_data/test_set/bitorsions_'


atom_missed_filenames = []
failed_filenames = []
filenames = os.listdir(PATH)
# for filename in [filenames[0]]:
for filename in filenames:
    print(filename)
    try:
        gt_coos = np.load(os.path.join(PATH, filename))
        if np.shape(gt_coos)[0] % 4 != 0:
            atom_missed_filenames.append(filename)

        else:
            torsions_C = []
            torsions_N = []
            coos = []
            for coo in gt_coos:
                coos.append(Coordinate(coo))

            for k in range(len(coos) // 4 - 2):
                k *= 4
                CA1 = coos[1 + k]
                C1 = coos[2 + k]
                O1 = coos[3 + k]
                N2 = coos[4 + k]
                CA2 = coos[5 + k]
                CA3 = coos[9 + k]

                CA2CA3 = vec(CA2, CA3)
                CA1CA2 = vec(CA1, CA2)
                CA2CA1 = vec(CA2, CA1)
                CA1C1 = vec(CA1, C1)
                CA2N2 = vec(CA2, N2)

                torsions_C.append(torsion(CA2CA3.orient, CA1C1.orient, CA1CA2.orient))
                if CA1CA2.len > 3.4:
                    torsions_N.append(angle_norm(torsion(CA2CA3.orient, CA2N2.orient, CA1CA2.orient) - math.pi))
                else:
                    torsions_N.append(torsion(CA2CA3.orient, CA2N2.orient, CA1CA2.orient))

            k = (len(coos) // 4 - 3) * 4
            CA1 = coos[1 + k]
            CA2 = coos[5 + k]
            C2 = coos[6 + k]
            O2 = coos[7 + k]
            N3 = coos[8 + k]
            CA3 = coos[9 + k]

            CA2CA3 = vec(CA2, CA3)
            CA3CA2 = vec(CA3, CA2)
            CA2CA1 = vec(CA2, CA1)
            CA2C2 = vec(CA2, C2)
            CA3N3 = vec(CA3, N3)

            torsions_C.append(torsion(CA2CA1.orient, CA2C2.orient, CA2CA3.orient))
            if CA2CA3.len > 3.4:
                torsions_N.append(angle_norm(torsion(CA2CA1.orient, CA3N3.orient, CA2CA3.orient) - math.pi))
            else:
                torsions_N.append(torsion(CA2CA1.orient, CA3N3.orient, CA2CA3.orient))

            torsions_N = np.array(torsions_N)
            torsions_C = np.array(torsions_C)
            bitorsions = np.array([torsions_C / math.pi,
                                   torsions_N / math.pi,
                                   np.sin(torsions_C),
                                   np.cos(torsions_C),
                                   np.sin(torsions_N),
                                   np.cos(torsions_N)]).astype('float32')
            np.save(os.path.join(BITORSION_PATH, filename), bitorsions)

            rebuild_coos = np.concatenate((gt_coos[:1],
                                           backbone_rebuild_separated_torsion(coos[1::4], torsions_C, torsions_N),
                                           gt_coos[-2:]), axis=0).astype('float32')
            np.save(os.path.join(REBUILD_PATH, filename), rebuild_coos)
            # print(np.shape(rebuild_coos), np.shape(gt_coos))
            # print(np.linalg.norm(rebuild_coos - gt_coos, axis=1))
            # print(np.linalg.norm(rebuild_coos[::4] - gt_coos[::4], axis=1).mean())
            # print(np.linalg.norm(rebuild_coos[2::4] - gt_coos[2::4], axis=1).mean())

    except Exception:
        failed_filenames.append(filename)
