import math
import os
import numpy as np
import xlrd
import xlwt
from numpy import *
import pandas as pd

#通过两坐标计算单位向量
def vector_unit(vector_1,vector_2):
    bond_vector_2 = vector_1 - vector_2
    bond_length_2 = np.linalg.norm(bond_vector_2)
    return bond_vector_2 / bond_length_2


#计算法向量和旋转角
def torsion_angle(A, B, C, D):
    #计算法向量
    U_2 = vector_unit(B,A); U_1 = vector_unit(C,B); U = vector_unit(D,C)
    N   = np.cross(U_1, U)   / np.linalg.norm(np.cross(U_1, U))
    N_1 = np.cross(U_2, U_1) / np.linalg.norm(np.cross(U_2, U_1))
    m_weight = np.array([U_1, np.cross(N_1, U_1), N_1])
    #torsion_angle
    angle = np.sign(np.dot(U_2, N)) * math.acos(np.dot(N_1, N))  
    return angle, m_weight


#根据真实角度或训练角度预测下一个坐标
def next_coord(A, B, C, D, R, angle_confirm,torsion_pred):
    #torsion_angle
    angle_real , m = torsion_angle(A, B, C, D)
    #将真实角度或预测角度赋值给torsion
    torsion = torsion_pred
#     print("N——angle:",angle_real,angle_train)
    angle_martix=[math.cos(math.pi-angle_confirm),
                  math.sin(math.pi-angle_confirm) * math.cos(torsion),
                  math.sin(math.pi-angle_confirm) * math.sin(torsion)]
    #计算下一个坐标
    next_corrd = C + R * np.dot(m.T, angle_martix)
    return next_corrd, torsion


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
    next_CB_coord, t = next_coord(CA_coord, N_coord, C, CB_coord, R, angle_confirm, torsion)
    return next_CB_coord


#从真实的pdb中提取对应原子坐标信息
def extract_info_from_pdb(path_file_real):
    delet = []
    # 获取真实pdb文件坐标
    f_real = open(path_file_real, 'r');
    real_lines = f_real.readlines()

    # 提取对应原子的信息存到列表real
    real = [line for line in real_lines if line.split()[0] == 'ATOM' and line.split()[2] in  atoms_type]
    f_real.close()

    # 筛掉重复概率小的氨基酸
    for i in range(len(real)):
        if real[i - 1].split()[2] == real[i].split()[2] and real[i - 1].split()[5] == real[i].split()[5]:
            if real[i - 1].split()[-3] <= real[i].split()[-3]:
                delet.append(i - 1)
            else:
                delet.append(i)
    for i in delet[::-1]:
        del real[i]
    return real


# 记录该氨基酸是否存在CB
def CB_determine(real):
    real_CB = []
    CB_whether_exist = []
    real_array_without_CB = []
    real_array = np.zeros((len(real), 3))
    # real_with_CB = np.zeros((len(real), 3))
    for i in range(len(real)):
        real_array[i] = np.array([float(real[i].split()[j]) for j in range(6, 9)])
        #为了判定该CA处是否存在CB
        if real[i].split()[2] == atoms_type[1]:
            for line in range(-1, len(atoms_type) - 2):
                real_array_without_CB.append([float(real[i + line].split()[j]) for j in range(6, 9)])

            if real[i].split()[3] == 'GLY':
                CB_whether_exist.append('-')
            else:
                CB_coord = np.array([float(real[i + len(atoms_type) - 2].split()[j]) for j in range(6, 9)])
                # 检查重构CB和真实CB的误差
                # next_CB = pred_CBcoord(N_coord, CA_coord, C_coord, CB_coord)
                real_CB.append(CB_coord)
                CB_whether_exist.append('+')
    real_array_without_CB = array(real_array_without_CB)

    return real_CB, CB_whether_exist, real_array, real_array_without_CB


#从预测的pdb中提取对应原子坐标信息
def extract_info_from_pred(CB_whether_exist, path_pred):
    gen = []
    length_file = CB_whether_exist.count('+') * len(atoms_type) + CB_whether_exist.count('-') * (len(atoms_type) - 1)
    all_atoms = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}
    if path_pred.endswith('.pdb'):
        # pdb格式读取
        path_pred = path_pred.replace('real', 'pd2_out')
        f_gen = open(path_pred, 'r')
        gen_lines = f_gen.readlines()

        for line in range(len(gen_lines)):
            if gen_lines[line].split()[0] == 'ATOM' and gen_lines[line].split()[2] == 'CB':
                gen.append(gen_lines[line])
        f_gen.close()
        # 提取全部坐标为array，提取CB坐标为array_CB
        pred_array = np.zeros((length_file, 3))
        pred_CB = np.zeros((CB_whether_exist.count('+'), 3))
        pred_array_without_CB = np.zeros((length_file - CB_whether_exist.count('+'), 3))
        count = 0; count_CB = 0
        print(length_file, len(gen), len(CB_whether_exist))
        for i in range(len(gen)):
            pred_array[i] = np.array([float(gen[i].split()[j]) for j in range(6, 9)])
            if gen[i].split()[2] == 'CB':
                pred_CB[count_CB] = np.array([float(gen[i].split()[j]) for j in range(6, 9)])
                count_CB += 1
            else:
                pred_array_without_CB[count] = np.array([float(gen[i].split()[j]) for j in range(6, 9)])
                count += 1
        return pred_array, pred_array_without_CB, pred_CB
    else:
        # npy格式读取
        path_pred = path_pred.replace('real', 'our_out')
        pred_array = []
        number = [all_atoms[atom] for atom in atoms_type[:-1]]
        pred_npy_without_CB = np.load(path_pred)
        pred_CB = np.zeros((CB_whether_exist.count('+'), 3))
        pred_array_without_CB = []
        count = 0
        for j in range(0, pred_npy_without_CB.shape[0]):
            # 进一步提取只有CB的array
            if j % 4 == int(number[-1]):
                for line in number:
                    pred_array.append(pred_npy_without_CB[j +line - number[-1]])
                    pred_array_without_CB.append(pred_npy_without_CB[j +line - number[-1]])

                if CB_whether_exist[j // 4] == '+':
                    N_coord_pred  = pred_npy_without_CB[j - number[-1]]
                    CA_coord_pred = pred_npy_without_CB[j - number[-1] +1]
                    C_coord_pred  = pred_npy_without_CB[j - number[-1] +2]
                    CB_coord_pred = pred_npy_without_CB[j - number[-1] +3]
                    #根据预测出的C和N计算得到CB的坐标
                    next_CB = pred_CBcoord(N_coord_pred, CA_coord_pred, C_coord_pred, CB_coord_pred)
                    pred_CB[count] = np.array(next_CB)
                    pred_array.append(next_CB)
                    count += 1
        pred_array = array(pred_array)
        pred_array_without_CB = array(pred_array_without_CB)
        return  pred_array, pred_array_without_CB, pred_CB


# pdb或npy数据的位置，
# 可以返回含有所有原子的array
# 含（N,CA,C,O)的array
# 只含CB的array
def extraction_coord(path_real, path_pred):
    # 获取真实pdb的坐标信息返回为array
    real = extract_info_from_pdb(path_real)
    real_CB, CB_whether_exist, real_array, real_array_without_CB = CB_determine(real)

    # 获取pred的坐标信息返回为array
    pred_array, pred_array_without_CB, pred_CB = extract_info_from_pred(CB_whether_exist, path_pred )

    return real_array, pred_array, real_array_without_CB, pred_array_without_CB, real_CB, pred_CB


def test(real,pred):
    GC = 0
    for i in range(len(real)):
        A = real[i]
        B = pred[i]
        GC += np.square(np.linalg.norm(np.array(A) - np.array(B)))
        print(A-B)


#real:真实的坐标数组 pred:生成的坐标数组
def computation_rmsd(real, pred):
    K = np.eye(4)
    Sxx = Sxy = Sxz = Syx = Syy = Syz = Szx = Szy = Szz = 0
    GA = GB = GC = 0

    for i in range(len(real)):
        A = real[i]; B = pred[i]
        XA = A[0]; YA = A[1]; ZA = A[2]
        XB = B[0]; YB = B[1]; ZB = B[2]

        GA += np.square(np.linalg.norm(A))
        GB += np.square(np.linalg.norm(B))
        GC += np.square(np.linalg.norm(np.array(A) - np.array(B)))

        Sxx += XB * XA; Syy += YB * YA; Szz += ZB * ZA
        Sxy += XB * YA; Sxz += XB * ZA; Syz += YB * ZA
        Syx += YB * XA; Szx += ZB * XA; Szy += ZB * YA

    # 构建密钥矩阵
    K[0][0] = Sxx + Syy + Szz
    K[1][1] = Sxx - Syy - Szz
    K[2][2] = -Sxx + Syy - Szz
    K[3][3] = -Sxx - Syy + Szz
    K[0][1] = K[1][0] = Syz - Szy
    K[0][2] = K[2][0] = Szx - Sxz
    K[0][3] = K[3][0] = Sxy - Syx
    K[1][2] = K[2][1] = Sxy + Syx
    K[1][3] = K[3][1] = Szx + Sxz
    K[2][3] = K[3][2] = Syz + Szy

    # 计算最大特征值
    a, b = np.linalg.eig(K)
    u = max(a)

    # 计算rmsd
    rmsd = np.sqrt(abs((GA + GB - 2 * u)) / len(real))
    # C_rmsd = np.sqrt(GC / len(real))
    return rmsd

#计算所有rmsd数值的矩阵
def computation_rmsd_array(pred_end, sheet):
    atoms = atoms_type[0:-1]
    # 获取需要计算的文件名
    file_names = os.listdir(os.path.join(os.getcwd(), 'real'))
    # 暂且不算有问题的pdb， 具体问题正在进一步查找
    # file_names.remove('4fbr.pdb')
    file_names.remove('4avz.pdb')

    # 遍历所有pdb文件
    path_file = os.path.join(os.getcwd(), 'real')
    rmsd_array = np.zeros((len(file_names), len(atoms) + 2))

    for file_real in file_names:
        idx = file_names.index(file_real) + 1
        sheet.write(idx, 0, file_real)

        path_real = os.path.join(path_file, file_real)
        if pred_end.endswith('pdb'):
            path_pred = path_real.replace('.pdb', '_out.pdb')
        elif pred_end.endswith('npy'):
            path_pred = path_real.replace('.pdb', '.npy')  # 想要和真实数据集进行对比的文件后缀

        # 计算每个原子的rmsd
        real, pred, real_without_CB, pred_without_CB, real_CB, pred_CB = extraction_coord(path_real, path_pred)

        for atom in atoms:
            row = atoms.index(atom)
            real_atom = real_without_CB[row::len(atoms_type)-1]
            pred_atom = pred_without_CB[row::len(atoms_type)-1]
            rmsd_atom = computation_rmsd(real_atom, pred_atom)
            rmsd_array[idx - 1][row] = rmsd_atom

        # 计算CB的rmsd
        rmsd_CB = computation_rmsd(real_CB, pred_without_CB)
        rmsd_array[idx - 1][row + 1] = rmsd_CB
        # 计算全原子的rmsd
        rmsd = computation_rmsd(real, pred)
        rmsd_array[idx - 1][row + 2] = rmsd

    sheet.write(idx + 1, 0, 'mean')
    means = np.mean(rmsd_array, axis=0)
    rmsd_array = np.insert(rmsd_array, rmsd_array.shape[0], values=means, axis=0)
    return rmsd_array

#将计算出的数值写入excel
if __name__ == "__main__":
    # 创建excel，并写入每列名称
    book = xlwt.Workbook(encoding="utf-8", style_compression=0)
    pred_end = input('请输入文件格式：')
    # Create a sheet object, a sheet object corresponding to a table in the Excel file.
    if pred_end.endswith('pdb'):
        sheet = book.add_sheet('PD2', cell_overwrite_ok=True)
    elif pred_end.endswith('npy'):
        sheet = book.add_sheet('our', cell_overwrite_ok=True)

    atoms_type = input('请按pdb原子排列顺序输入需要计算的原子(逗号隔开）：')
    if atoms_type == '':
        atoms_type= ['N', 'CA', 'C', 'O', 'CB']
    else:
        atoms_type = atoms_type.split(",")
    # 写入每一列的title
    names = ['file_name'] + atoms_type + ['scut']
    for i in range(len(names)):
        sheet.write(0, i, names[i])

    #获取所有计算数值的rmsd
    rmsd_array = computation_rmsd_array(pred_end, sheet)

    #将矩阵数据写入excel
    for i in range(rmsd_array.shape[0]):
        for j in range(rmsd_array.shape[1]):
            sheet.write(i+1, j+1, rmsd_array[i][j])

    if pred_end.endswith('pdb'):
        book.save('D://database//rmsd_compare//backbone_PD2.xls')
    elif pred_end.endswith('npy'):
        book.save('D://database//rmsd_compare//backbone1_our.xls')



