import os
# import xlrd
# from xlwt import Workbook
import argparse

import random
import numpy as np


def find_start_seq_npy_L_F_J(Dir):
    top = 40
    top_elements = np.zeros((top,3))
    num_of_rep = 0
    directory = os.fsencode(Dir + '/lengths')
    for f, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith('.npy'):
            num_of_rep += 1
            L = np.load(Dir + '/lengths/' + filename)
            vF = np.load(Dir + '/families/' + filename)
            # V = np.load(Dir + '/Vgenes/' + filename)
            J = np.load(Dir + '/Jgenes/' + filename)

            L_F_J = np.stack((L, vF, J), axis=1)

            # e, c = np.unique(J, axis=0, return_counts=True)
            # pp = (e).argsort()
            # print(e[pp])

            unique_elements, counts_elements = np.unique(L_F_J, axis=0, return_counts=True)
            p = (-counts_elements).argsort()
            sorted_unique_elements = unique_elements[p]
            sorted_counts_elements = counts_elements[p]

            top_elements = np.concatenate((top_elements, sorted_unique_elements[0:top]), axis=0)

    # find top of top
    unique_elements, counts_elements = np.unique(top_elements, axis=0, return_counts=True)
    p = (-counts_elements).argsort()
    sorted_unique_elements = unique_elements[p]
    sorted_counts_elements = counts_elements[p]
    print("num of rep:  ", num_of_rep)
    print("top5:  ", sorted_counts_elements[0:5])
    print("top element:  ", sorted_unique_elements[0])

    return sorted_unique_elements[0]


def find_start_seq_representative_npy_L_F_J(Dir, start_seq):
    final_representative = np.zeros((84,4))
    z = np.zeros((84,4))
    directory = os.fsencode(Dir + '/cubes/train')
    for f, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith('.npy'):
            data = np.load(Dir + '/cubes/train/' + filename)
            L = np.load(Dir + '/lengths/' + filename)
            vF = np.load(Dir + '/families/' + filename)
            # V = np.load(Dir + '/Vgenes/' + filename)
            J = np.load(Dir + '/Jgenes/' + filename)
            L_F_J = np.stack((L, vF, J), axis=1)

            temp_representative = np.zeros((84,4))
            for i in range(len(L)):
                if(np.array_equal(L_F_J[i], start_seq)):
                    temp_representative += data[i]

            z[np.arange(len(z)), np.argmax(temp_representative, axis=-1)] += 1
    final_representative[np.arange(len(final_representative))[0:48], np.argmax(z, axis=-1)[0:48]] += 1


    filename = '/cubes/start_seq_representative.npy'
    np.save(Dir + filename, final_representative)


def find_start_seq(Dir):
    top = 5
    top_elements = np.zeros((top,3))
    num_of_rep = 0
    directory = os.fsencode(Dir)
    for f, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith('.xls'):
            num_of_rep += 1
            wb = xlrd.open_workbook(Dir + '/' + filename)
            sheet = wb.sheet_by_index(0)
            rep = np.zeros((sheet.nrows - 1, 3))
            for i in range(sheet.nrows - 1):
                #len
                rep[i,0] = int(sheet.cell_value(i, 0))
                #V
                v = sheet.cell_value(i, 2)
                start = v.find("-") + 1
                if(v[start + 1].isnumeric()):
                    rep[i,1] = int(v[start:start + 2])
                elif(v[start].isnumeric()):
                    rep[i,1] = int(v[start:start + 1])
                else:
                    rep[i,1] = 0
                #J
                j = sheet.cell_value(i, 3)
                rep[i,2] = int(j[4:5])

            unique_elements, counts_elements = np.unique(rep, axis=0, return_counts=True)
            p = (-counts_elements).argsort()
            sorted_unique_elements = unique_elements[p]
            sorted_counts_elements = counts_elements[p]

            top_elements = np.concatenate((top_elements, sorted_unique_elements[0:top]), axis=0)

    # find top of top
    unique_elements, counts_elements = np.unique(top_elements, axis=0, return_counts=True)
    p = (-counts_elements).argsort()
    sorted_unique_elements = unique_elements[p]
    sorted_counts_elements = counts_elements[p]
    print("num of rep:  ", num_of_rep)
    print("top5:  ", sorted_counts_elements[0:5])
    print("top element:  ", sorted_unique_elements[0])
    print("top element:  ", sorted_unique_elements[1])

    return sorted_unique_elements[0]

def find_start_seq_representative(Dir, start_seq):
    l, v, j = start_seq
    all_representative = np.zeros((1,l)) + 7
    directory = os.fsencode(Dir)
    for f, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith('.xls'):
            wb = xlrd.open_workbook(Dir + '/' + filename)
            sheet = wb.sheet_by_index(0)
            frequencies_in_rep = np.zeros((l,4))
            for i in range(sheet.nrows - 1):
                # len
                temp_l = int(sheet.cell_value(i, 0))
                # v_call
                v_call = sheet.cell_value(i, 2)
                start = v_call.find("-") + 1
                if (v_call[start + 1].isnumeric()):
                    temp_v = int(v_call[start:start + 2])
                elif (v_call[start].isnumeric()):
                    temp_v = int(v_call[start:start + 1])
                else:
                    temp_v = 0
                # j_call
                j_call = sheet.cell_value(i, 3)
                temp_j = int(j_call[4:5])

                if((temp_l == l) & (temp_v == v) & (temp_j == j)):
                    temp_seq = sheet.cell_value(i, 1)
                    for i, char in enumerate(temp_seq):
                        if char == 'A':
                            frequencies_in_rep[i, 0] += 1
                        elif char == 'C':
                            frequencies_in_rep[i, 1] += 1
                        elif char == 'G':
                            frequencies_in_rep[i, 2] += 1
                        elif char == 'T':
                            frequencies_in_rep[i, 3] += 1

            rep_representative = np.argmax(frequencies_in_rep, axis=1)
            all_representative = np.concatenate((all_representative, [rep_representative]), axis=0)

    representative = np.zeros((84,4))
    for i in range(l):
        unique_elements, counts_elements = np.unique(all_representative[:,i], axis=0, return_counts=True)
        p = (-counts_elements).argsort(axis=0)
        c = unique_elements[p][0]
        if c == 0:
            representative[i, 0] = 1
        elif c == 1:
            representative[i, 1] = 1
        elif c == 2:
            representative[i, 2] = 1
        elif c == 3:
            representative[i, 3] = 1

    filename = '/start_seq_representative.npy'
    np.save(Dir + filename, representative)
    print('start_seq_representative ready')




# default_data_path = r'C:\Users\user\Desktop\limudim\M.Sc\RepertoiresClassification\randomNsec\Data\celiac_data\XL'
default_data_path = r'C:\Users\user\Desktop\limudim\M.Sc\RepertoiresClassification\datasets\celiac_data\XL\Cleaned'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data_path',
                        help='data_path',
                        type=str,
                        default=default_data_path)
    args = parser.parse_args()

    print('find_start_seq_npy_L_F_J')
    start_seq = find_start_seq_npy_L_F_J(args.data_path)
    print('find_start_seq_representative_npy_L_F_J')
    find_start_seq_representative_npy_L_F_J(args.data_path, start_seq)

    # start_seq = find_start_seq(args.data_path)
    # start_seq = np.array([48, 23, 4])   # [length, Vgene, Jgene]
    # find_start_seq_representative(args.data_path, start_seq)
