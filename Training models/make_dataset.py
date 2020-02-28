# -*- coding: utf-8 -*
import os
import time
import pathlib
import argparse
from process import Process

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--resolution', type=int, default='3000',
                    help='output resolution')
parser.add_argument('--dataset_path', type=str, default='D:\protein_structure_prediction\data\dataset/casp12_sep',
                    help='path of dataset')
parser.add_argument('--output_path', type=str, default='D:\protein_structure_prediction\data\dataset\processed_data',
                    help='path of output')
parser.add_argument('--dataset', type=str, default='training_95',
                    help='name of dataset folder, training_95|bc-30-1_CA|bc-30-1_chains|cif_filtered|cif_fragment|nr90')
parser.add_argument('--input_type', type=str, default='pn',
                    help='type of input file, pn|cif|pdb')
parser.add_argument('--output_type', type=str, default='image',
                    help='output_format, images|distance_map|relocated_coordinate')
parser.add_argument('--axis_range', type=int, default='100',
                    help='map range of structures, 42|64|84')
parser.add_argument('--multi_process', type=bool, default=False,
                    help='multi process or not')
parser.add_argument('--multi_atom', type=bool, default=False,
                    help='input all backbone atoms or CA only')
parser.add_argument('--self_norm_ser_num', type=bool, default=False,
                    help='self normalized serial number')
parser.add_argument('--draw_connection', type=bool, default=True,
                    help='draw dots connection or not')
parser.add_argument('--crop', type=bool, default=True,
                    help='crop image before output')
parser.add_argument('--aminoacid_message', type=bool, default=True,
                    help='mark amino acid with hydropathicity, bulkiness and flexibility or 1.')
parser.add_argument('--z_norm', type=float, default=64.,
                    help='normalize range of z value')
parser.add_argument('--pairs_data', action='store_true', default=False,
                    help='pairs_data')
parser.add_argument('--test', action='store_true', default=True,
                    help='test mode')
parser.add_argument('--filenames_list', type=str, default='validation_len_under_200.txt',
                    help='read input filenames in list')
parser.add_argument('--sliding_window', action='store_true', default=True,
                    help='save outputs as sliding window')
parser.add_argument('--window_reorient', action='store_true', default=True,
                    help='reorientation for normalize every sliding window')
argparses = parser.parse_args()


class MakeDataset(object):
    def __init__(self, args):
        self.args = args
        self.input_folder = os.path.join(args.dataset_path, args.dataset)
        if args.filenames_list:
            with open(os.path.join(args.dataset_path, args.filenames_list), 'r') as file:
                self.filenames = file.read().split('\n')
        else:
            self.filenames = os.listdir(self.input_folder)
        self.output_folders = {}

    def run(self):
        output_folder = os.path.join(self.args.output_path, self.args.dataset, time.strftime("%Y%m%d_%H%M",
                                                                                             time.localtime()))
        log_folder = os.path.join(self.args.output_path, self.args.dataset)
        self.make_folders(output_folder)
        self.write_log(log_folder)
        for filename in self.filenames:
            Process(self.args, filename, self.output_folders)

    def test(self, sample_num=5):
        output_folder = self.args.output_path + '/test_sample'
        self.make_folders(output_folder)
        self.write_log(output_folder)
        for filename in ['4KE2_1_A.pn']:  # self.filenames[:sample_num]:
            Process(self.args, filename, self.output_folders).process_for_data_loader_test()

    def make_folders(self, output_folder):
        self.output_folders.update({'output': output_folder})
        if self.args.pairs_data:
            query_folder = output_folder + '/query'
            target_folder = output_folder + '/target'
            pathlib.Path(query_folder).mkdir(parents=True, exist_ok=True)
            pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)
            self.output_folders.update({'query': query_folder, 'target': target_folder})
        else:
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    def write_log(self, path):
        args = self.args
        write_list = [time.strftime("%Y%m%d_%H%M", time.localtime())]
        arg_name_list = ['dataset',
                         'resolution',
                         'input_type',
                         'output_type',
                         'axis_range',
                         'multi_atom',
                         'self_norm_ser_num',
                         'draw_connection',
                         'z_norm']
        arg_list = [args.dataset,
                    args.resolution,
                    args.input_type,
                    args.output_type,
                    args.axis_range,
                    args.multi_atom,
                    args.self_norm_ser_num,
                    args.draw_connection,
                    args.z_norm]
        for i in range(len(arg_name_list)):
            print("%s = %s" % (arg_name_list[i], str(arg_list[i])))
            write_list.append("%s = %s" % (arg_name_list[i], str(arg_list[i])))
        write_list.append('\n\n\n')
        with open(path + '/args_log.txt', 'a') as log_writer:
            log_writer.write('\n'.join(write_list))


if __name__ == '__main__':
    if argparses.test:
        MakeDataset(argparses).test()
    else:
        MakeDataset(argparses).run()
