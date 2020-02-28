import numpy as np
import math
import os
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pathlib
import matplotlib.pyplot as plt

L1_C = 0.5511235634596036
L1_N = 0.5275157666844377
# trans
L2_C_trans = 1.4281242923706199
R_C_trans = 0.5298886988235514
L2_N_trans = 1.4076846053568244
R_N_trans = 0.3797594132360668
L_O_trans = 1.669968615090273
R_O_trans = 1.735878468087069
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
    return Coordinate(b.coo - a.coo)


def get_coo(line):
    items = line.split()
    x = float(items[10])
    y = float(items[11])
    z = float(items[12])
    return Coordinate(np.array([x, y, z]))


def get_coos(lines):
    atom_coos = []
    for line in lines:
        atom_coos.append(get_coo(line))
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
    return np.dot(cb.coo, cd.coo)/(cb.len * cd.len)


def get_angle(cb, cd):
    return math.acos(get_cos(cb, cd))


def angle_norm(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def batch_angle_norm(array):
    return np.arctan2(np.sin(array), np.cos(array))


def get_projection(vector, axis):
    return Coordinate(vector.len * get_cos(vector, axis) * axis.orient)


def get_sign(vector, axis):
    return Coordinate(vector.coo - get_projection(vector, axis).coo)


# 计算以axis为轴，向量A到向量B的旋转角
def get_torsion(vector_A, vector_B, axis):
    N = Coordinate(np.cross(axis, vector_B))
    N_1 = Coordinate(np.cross(vector_A, axis))
    torsion = np.sign(np.dot(vector_A, N.orient)) * math.acos(np.dot(N_1.orient, N.orient))
    return torsion


def distance_martix(coordinates):
    return squareform(pdist(coordinates, metric='euclidean'))


# 计算夹角和坐标转换权重
def torsion_m(vector_A, axis):
    # 计算法向量
    N_1 = Coordinate(np.cross(vector_A, axis)).orient
    # 旋转基向量
    m_weight = np.array([axis, np.cross(N_1, axis), N_1])
    angle = math.acos(np.dot(axis, vector_A))
    return m_weight, angle


# 根据向量，旋转轴 旋转角 计算旋转过后的向量
def rotation(vector_A, axis, torsion):
    m, angle = torsion_m(vector_A, axis)
    rotation_martix = [math.cos(math.pi-angle),
                       math.sin(math.pi-angle) * math.cos(torsion),
                       math.sin(math.pi-angle) * math.sin(torsion)]

    # 计算旋转后向量
    vector_B = np.dot(m.T, rotation_martix)
    return vector_B


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
