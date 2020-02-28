import math
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
#from scipy.stats import norm as nm
from multiprocessing import Pool
import argparse
import matplotlib.image
# from .arraylize import Arraylize


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--resolution', type=int, default='256',
                    help='output resolution')
parser.add_argument('--dataset_path', type=str, default=os.getcwd(),
                    help='path of dataset')
parser.add_argument('--output_path', type=str, default=os.getcwd()+'/processed_data',
                    help='path of output')
parser.add_argument('--dataset', type=str, default='cif_filtered',
                    help='name of dataset folder, bc-30-1_CA|bc-30-1_chains|cif_filtered')
parser.add_argument('--input_type', type=str, default='cif',
                    help='type of input file, cif|pdb')
parser.add_argument('--output_type', type=str, default='image',
                    help='image or distance_map, images|distance_map')
parser.add_argument('--axis_range', type=int, default='64',
                    help='map range of structures, 42|64')
parser.add_argument('--multi_process', type=bool, default=True,
                    help='multi process or not')
parser.add_argument('--multi_atom', type=bool, default=False,
                    help='input all backbone atoms or CA only')
parser.add_argument('--move2center', type=bool, default=True,
                    help='relocate the center of proteins to the center of coordinate system')
parser.add_argument('--redistribute', type=bool, default=False,
                    help='redistribute the original distribution according to normal distribution')
parser.add_argument('--relative_number', type=bool, default=False,
                    help='mark dots with relative serial number')
parser.add_argument('--draw_connection', type=bool, default=True,
                    help='draw dots connection or not')
parser.add_argument('--aminoacid_message', type=bool, default=True,
                    help='mark amino acid with hydropathicity, bulkiness and flexibility or 1.')
parser.add_argument('--redistribute_rate', type=float, default='1.4',
                    help='coefficient of redistribution amplitude')
args = parser.parse_args()

res = args.resolution
ar = args.axis_range
s = ar / res  # scale=axis_range/resolution
input_folder = args.dataset_path + '/' + args.dataset
AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
               'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO',
               'SER', 'THR', 'TRP', 'TYR', 'VAL']
AA_HYDROPATHICITY_INDEX = {
    'ARG': -4.5,
    'LYS': -3.9,
    'ASN': -3.5,
    'ASP': -3.5,
    'GLN': -3.5,
    'GLU': -3.5,
    'HIS': -3.2,
    'PRO': -1.6,
    'TYR': -1.3,
    'TRP': -0.9,
    'SER': -0.8,
    'THR': -0.7,
    'GLY': -0.4,
    'ALA': 1.8,
    'MET': 1.9,
    'CYS': 2.5,
    'PHE': 2.8,
    'LEU': 3.8,
    'VAL': 4.2,
    'ILE': 4.5,
}
AA_BULKINESS_INDEX = {
    'ARG': 14.28,
    'LYS': 15.71,
    'ASN': 12.82,
    'ASP': 11.68,
    'GLN': 14.45,
    'GLU': 13.57,
    'HIS': 13.69,
    'PRO': 17.43,
    'TYR': 18.03,
    'TRP': 21.67,
    'SER': 9.47,
    'THR': 15.77,
    'GLY': 3.4,
    'ALA': 11.5,
    'MET': 16.25,
    'CYS': 13.46,
    'PHE': 19.8,
    'LEU': 21.4,
    'VAL': 21.57,
    'ILE': 21.4,
}
AA_FLEXIBILITY_INDEX = {
    'ARG': 2.6,
    'LYS': 1.9,
    'ASN': 14.,
    'ASP': 12.,
    'GLN': 4.8,
    'GLU': 5.4,
    'HIS': 4.,
    'PRO': 0.05,
    'TYR': 0.05,
    'TRP': 0.05,
    'SER': 19.,
    'THR': 9.3,
    'GLY': 23.,
    'ALA': 14.,
    'MET': 0.05,
    'CYS': 0.05,
    'PHE': 7.5,
    'LEU': 5.1,
    'VAL': 2.6,
    'ILE': 1.6,
}
AMINO_ACID_NUMBERS = {}
if args.aminoacid_message:
    for aa in AMINO_ACIDS:
        AMINO_ACID_NUMBERS.update({aa: [(5.5-AA_HYDROPATHICITY_INDEX[aa]) / 10 * 255.,
                                        AA_BULKINESS_INDEX[aa] / 21.67 * 255.,
                                        (25.-AA_FLEXIBILITY_INDEX[aa]) / 25. * 255.]})
else:
    for aa in AMINO_ACIDS:
        AMINO_ACID_NUMBERS.update({aa: [1.]})
ary_dim = 2 + len(AMINO_ACID_NUMBERS[AMINO_ACIDS[0]])


class Atom(object):
    def __init__(self, aminoacid, index, x, y, z, atom_type='CA', element='C'):
        self.index = int(index)
        self.aa = aminoacid
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.type = atom_type
        self.element = element


def readfile(filename, path):
    file = open(path + '/' + filename, 'r')
    if os.path.splitext(filename)[1] == '.cif'  or os.path.splitext(filename)[1]=='.pdb':
        message = file.readlines()
        return message

    file.close()



def extract_cif(cif_message):
    atoms = []
    for line in cif_message:
        line = line.split()
        if line[3] in ['CA', 'C', 'N']:
            atoms.append(Atom(line[5], line[8], line[10],
                              line[11], line[12], line[3], line[2]))
    return atoms


def extract_ca_cif(cif_message):
    atoms = []
    for line in cif_message:
        line = line.split()
        if line[3] == 'CA':
            atoms.append(Atom(line[5], line[8], line[10], line[11], line[12]))
    return atoms


def extract_pdb(pdb_message):
    atoms = []
    for line in pdb_message:
        if line[13:15] in ['N ', 'CA', 'C ']:
            atoms.append(Atom(line[17:20], line[13:16], line[30:38],
                              line[38:46], line[46:54], line[13:16], line[77]))
    return atoms


def extract_ca_pdb(pdb_message):
    atoms = []
    for line in pdb_message:
        if line[13:15] == 'CA':
            atoms.append(Atom(line[17:20], line[13:16], line[30:38], line[38:46], line[46:54]))
    return atoms


def extract_message(message, message_type):
    if message_type == 'pdb':
        if args.multi_atom:
            return extract_pdb(message)
        else:
            return extract_ca_pdb(message)
    elif message_type == 'cif':
        if args.multi_atom:
            return extract_cif(message)
        else:
            return extract_ca_cif(message)


def find_head(atoms):
    for atom in atoms:
        if atom.type == 'CA':
            return atom


def find_tail(atoms):
    for i in range(1, len(atoms)+1):
        if atoms[-i].type == 'CA':
            return atoms[-i]


def rotation_axis(head):
    x = head.x
    y = head.y
    z = head.z
    c = ((y - x) ** 2 /
         ((y * res * (x ** 2 + y ** 2 + z ** 2 - 2 * s ** 2) ** 0.5 / ar - z) ** 2
          + (x * res * (x ** 2 + y ** 2 + z ** 2 - 2 * s ** 2) ** 0.5 / ar - z) ** 2
          + (y - x) ** 2)
         ) ** 0.5
    a = (y * res * (x ** 2 + y ** 2 + z ** 2 - 2 * s ** 2) ** 0.5 / ar - z) / (x - y) * c
    b = (x * res * (x ** 2 + y ** 2 + z ** 2 - 2 * s ** 2) ** 0.5 / ar - z) / (y - x) * c
    return [(a, b, c), (-a, -b, -c)]  # 转轴


def rotation_angle(head):
    x = head.x
    y = head.y
    z = head.z
    return math.acos(
        ((x + y) * s + z * (x ** 2 + y ** 2 + z ** 2 - 2 * s ** 2) ** 0.5) /
        (x ** 2 + y ** 2 + z ** 2)
    )  # 转角


def rotation(u, v, w, t, axis):  # 原始坐标
    (a, b, c) = axis
    # 罗德里格旋转公式：
    rx = u*math.cos(t)+(b*w-c*v)*math.sin(t)+a*(a*u+b*v+c*w)*(1-math.cos(t))
    ry = v*math.cos(t)+(c*u-a*w)*math.sin(t)+b*(a*u+b*v+c*w)*(1-math.cos(t))
    rz = w*math.cos(t)+(a*v-b*u)*math.sin(t)+c*(a*u+b*v+c*w)*(1-math.cos(t))
    return rx, ry, rz  # 旋转所得坐标


def relocate(atoms):
    head = find_head(atoms)
    tail = find_tail(atoms)
    x_o = (head.x + tail.x) / 2
    y_o = (head.y + tail.y) / 2
    z_o = (head.z + tail.z) / 2
    for atom in atoms:
        atom.x -= x_o
        atom.y -= y_o
        atom.z -= z_o
    vs = rotation_axis(head)
    t = rotation_angle(head)
    atom_v = []
    for v in vs:
        atom_v.append(rotation(head.x, head.y, head.z, t, v))
    if abs(atom_v[0][0] - s) + abs(atom_v[0][1] - s) < abs(atom_v[1][0] - s) + abs(atom_v[1][1] - s):
        for atom in atoms:
            (atom.x, atom.y, atom.z) = rotation(atom.x, atom.y, atom.z, t, vs[0])
    else:
        for atom in atoms:
            (atom.x, atom.y, atom.z) = rotation(atom.x, atom.y, atom.z, t, vs[1])
    return atoms


def move2center(atoms):
    coordinates = []
    for atom in atoms:
        if atom.type == 'CA':
            coordinates.append([atom.x, atom.y, atom.z])
    coordinates = np.array(coordinates)
    center = tf.Variable(tf.zeros([1, 3]))
    distances = coordinates-center
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(distances), 1)))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    losses = []
    for step in range(10):
        sess.run(train)
        losses.append(sess.run(loss))
    while losses[-1] != losses[-5]:
        sess.run(train)
        losses.append(sess.run(loss))
    final_center = sess.run(center)[0]
    for atom in atoms:
        atom.x -= final_center[0]
        atom.y -= final_center[1]
        atom.z -= final_center[2]
    tf.reset_default_graph()
    return atoms


def sign(x):
    if x < 0:
        return -1
    else:
        return 1


def close_neibor(array, x_ary, y_ary, dot, dis_x, dis_y, rec):
    x_step = sign(dis_x)
    y_step = sign(dis_y)
    if abs(dis_x) < abs(dis_y):
        neibors = [(0, y_step), (x_step, 0), (x_step, y_step), (-x_step, 0),
                   (0, -y_step), (-x_step, y_step), (x_step, -y_step), (-x_step, -y_step)]
    else:
        neibors = [(x_step, 0), (0, y_step), (x_step, y_step), (0, -y_step),
                   (-x_step, 0), (x_step, -y_step), (-x_step, y_step), (-x_step, -y_step)]
    step = 1
    while True:
        for (i, j) in neibors:
            try:
                if array[x_ary + i * step, y_ary + j * step, 2] == 0:
                    array[x_ary + i * step, y_ary + j * step] = [dot.z, dot.index] + AMINO_ACID_NUMBERS.get(dot.aa)
                    rec.update({(x_ary + i * step, y_ary + j * step): dot})
                    # print('dot%d:%d,%d->%d,%d'%(dot[6],x,y,x+i*step,y+j*step))
                    return array
            except IndexError:
                print('dot(%d+%d,%d+%d) is out of the edge' % (x_ary, i * step, y_ary, j * step))
        # print('%d step neibor of dot%d(%d,%d) is full!'%(step,dot_i,x,y))
        step += 1


def lattice_battle(array, x_ary, y_ary, dot1, dot2, rec):  # dot1 is original; dot2 is new
    dis1_x = dot1.x / (2 * s) % 1 - 0.5
    dis1_y = dot1.y / (2 * s) % 1 - 0.5
    dis2_x = dot2.x / (2 * s) % 1 - 0.5
    dis2_y = dot2.y / (2 * s) % 1 - 0.5
    if dis1_x ** 2 + dis1_y ** 2 > dis2_x ** 2 + dis2_y ** 2:
        # print('%d / %d swap!'%(dot1[6],dot2[6]))
        array = close_neibor(array, x_ary, y_ary, dot1, dis1_x, dis1_y, rec)
        array[x_ary, y_ary] = [dot2.z, dot2.index] + AMINO_ACID_NUMBERS[dot2.aa]
        rec.update({(x_ary, y_ary) : dot2})
    else:
        array = close_neibor(array, x_ary, y_ary, dot2, dis2_x, dis2_y, rec)
    return array


def draw_atom(x, y, dot, array, rec):
    if array[x, y, -1] == 0:
        array[x, y] = [dot.z, dot.index] + AMINO_ACID_NUMBERS[dot.aa]
        rec.update({(x, y): dot})


def arraylize(atoms, array_dim):
    array = np.zeros([res, res, array_dim], dtype=float, order='C')
    rec = {}  # atoms record
    for atom in atoms:
        x_ary = int((atom.x + ar) // (2 * s))
        y_ary = int((atom.y + ar) // (2 * s))
        if rec.get((x_ary, y_ary)):
            array = lattice_battle(array, x_ary, y_ary, rec[(x_ary, y_ary)], atom, rec)
        else:
            draw_atom(x_ary, y_ary, atom, array, rec)
    return array, rec


# def values_sta(path):
#     xs = []
#     ys = []
#     for filename in os.listdir(path):
#         atoms = move2center(relocate(extract_cif(readfile(filename, path))))
#         for atom in atoms:
#             xs.append(atom.x)
#             ys.append(atom.y)
#     return xs, ys


def normal_dis(values, var, coefficient):
    dis = []
    values.sort()
    mark = 0
    idx = 0
    for i in range(res):
        cut_point = nm.ppf((i + 1) / res, 0, var**0.5 * coefficient)
        if idx == len(values):
            dis.append([])
            mark = int(idx)
        else:
            while values[idx] < cut_point:
                idx += 1
                if idx == len(values):
                    dis.append(values[mark:idx])
                    mark = int(idx)
                    break
            else:
                dis.append(values[mark:idx])
                mark = int(idx)
    return dis


# def redistribute():


def visual_values_dis(values):
    mark = 0
    idx = 0
    dis = []
    dis_count = []
    axis_length = 2*ar
    for i in range(1, res+1):
        cut_point = (i-res/2)*axis_length/res
        if idx == len(values):
            dis.append([])
        else:
            while values[idx] < cut_point:
                idx += 1
                if idx == len(values):
                    dis.append(values[mark:idx])
                    break
            else:
                dis.append(values[mark:idx])
                mark = int(idx)
    for i in range(res):
        dis_count.append(len(dis[i]))
    plt.bar(range(res), dis_count)
    plt.show()


def vis_normal_dis(values, var, coefficient):
    dis = []
    values.sort()
    mark = 0
    idx = 0
    dis_count = []
    for i in range(res):
        cut_point = nm.ppf((i+1)/res, 0, var**0.5*coefficient)
        if idx == len(values):
            dis.append([])
            mark = int(idx)
        else:
            while values[idx] < cut_point:
                idx += 1
                if idx == len(values):
                    dis.append(values[mark:idx])
                    mark = int(idx)
                    break
            else:
                dis.append(values[mark:idx])
                mark = int(idx)
        dis_count.append(len(dis[i]))
    plt.bar(range(res), dis_count)
    plt.show()


def draw_dot(x, y, dot1, z_add, idx_add, array):
    if array[x, y, 2] == 0:
        array[x, y] = [dot1.z + z_add, dot1.index + idx_add, 0, 0, 0]


def dots_connection(dot1, dot2, array, site):
    x = site[dot1][0]
    y = site[dot1][1]
    z_s = dot2.z - dot1.z
    x_r = sign(site[dot2][0] - x)
    y_r = sign(site[dot2][1] - y)
    x_s = abs(site[dot2][0] - x)
    y_s = abs(site[dot2][1] - y)
    dis_c = max(x_s, y_s)+1
    if x_s + y_s > 2:
        for i in range(max(x_s, y_s)):
            l = i + 1
            if min(x_s, y_s) <= 1:
                if x_s > y_s:
                    draw_dot(x + l*x_r, y, dot1, z_s*l/dis_c, l/dis_c, array)
                else:
                    draw_dot(x, y + l*y_r, dot1, z_s*l/dis_c, l/dis_c, array)
            else:
                t = max(x_s, y_s) // min(x_s, y_s)
                remainder = max(x_s, y_s) % min(x_s, y_s)
                if x_s > y_s:
                    j = [l, i//t, l, y_s]
                else:
                    j = [i//t, l, x_s, l]
                if i < max(x_s, y_s) - remainder:
                    draw_dot(x + j[0] * x_r, y + j[1] * y_r, dot1, z_s*l/dis_c, l/dis_c, array)
                else:
                    draw_dot(x + j[2] * x_r, y + j[3] * y_r, dot1, z_s*l/dis_c, l/dis_c, array)


def draw_connection(atoms, array, rec):
    site = {}
    for (x, y) in rec.keys():
        site.update({rec[(x, y)]: [x, y]})
    for i in range(len(atoms) - 1):
        dots_connection(atoms[i], atoms[i + 1], array, site)


def write_log(path):
    arg_name_list = ['dataset', 'resolution', 'input_type', 'output_type', 'axis_range', 'multi_atom',
                     'move2center', 'redistribute', 'redistribute_rate', 'relative_number', 'draw_connection',
                     'aminoacid_message']
    arg_list = [args.dataset, args.resolution, args.input_type, args.output_type, args.axis_range, args.multi_atom,
                args.move2center, args.redistribute, args.redistribute_rate, args.relative_number, args.draw_connection,
                args.aminoacid_message]
    write_list = [time.strftime("%Y%m%d_%H%M", time.localtime())]
    for i in range(len(arg_name_list)):
        print("%s = %s" % (arg_name_list[i], str(arg_list[i])))
        write_list.append("%s = %s" % (arg_name_list[i], str(arg_list[i])))
    write_list.append('\n\n\n')
    with open(path + '/args_log.txt', 'a') as log_writer:
        log_writer.write('\n'.join(write_list))


def process():
    log_dir = args.output_path + '/' + args.dataset
    output_dir = args.output_path + '/' + args.dataset + '/' + time.strftime("%Y%m%d_%H%M", time.localtime())
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_log(log_dir)
    num = 0
    if args.output_type == 'image':
        if args.redistribute:
            atoms_dic = {}
            xs = []
            ys = []
            for filename in os.listdir(input_folder):
                atoms = relocate(extract_message(readfile(filename, input_folder), args.input_type))
                if args.move2center:
                    atoms = move2center(atoms)
                for atom in atoms:
                    xs.append(atom.x)
                    ys.append(atom.y)
                atoms_dic.update({filename: atoms})
            # var_sta = max(np.var(xs), np.var(ys))
        else:
            for filename in os.listdir(input_folder):
                atoms = relocate(extract_message(readfile(filename, input_folder), args.input_type))
                if args.move2center:
                    atoms = move2center(atoms)
                    for i in range(len(atoms)):
                        atoms[i].z=(atoms[i].z+64.)*2.-2.
                if args.draw_connection:
                    array, rec = arraylize(atoms, ary_dim)
                    draw_connection(atoms, array, rec)
                else:
                    array, _ = arraylize(atoms, ary_dim)
                if args.relative_number:
                    array[:, :, 1] /= (len(atoms) + 1)
                output_name = filename.replace('.cif', '.npy')

                np.save(output_dir + '/' + output_name, array)
                # break
                # matplotlib.image.imsave(output_dir + '/' + output_name.replace('.npy', '.png'), array)
                # num += 1
                # if num == 10:
                #     break
    elif args.output_type == 'distance_map':
        if args.multi_atom:
            for filename in os.listdir(input_folder):
                atoms = extract_message(readfile(filename, input_folder), args.input_type)


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(3)
    p.apply_async(process())
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

