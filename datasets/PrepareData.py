import os
import random
import xlrd
import numpy as np
import argparse
from xlwt import Workbook
import csv
import time
import ray
from pathlib import Path
from l_f_j_v_lists import all_j_f_v_l_percent, true_clusters_1, true_dic, false_dic, test_list

# ray.init(memory=60000000000, object_store_memory=60000000000)
# ray.init()

cubeSize = 1000
numOfReps = 6

main_dir = os.getcwd()

exclude_list = list(['HIP00614', 'HIP00710', 'HIP00779', 'HIP00951', 'HIP01181', 'HIP01298', 'HIP01359', 'HIP01501', 'HIP01765', 'HIP01867', 'HIP03385', 'HIP03484', 'HIP03685', 'HIP04455', 'HIP04464', 'HIP05552', 'HIP05757', 'HIP05934', 'HIP05941', 'HIP05948', 'HIP08339', 'HIP08596', 'HIP08821', 'HIP09681', 'HIP10424', 'HIP10815', 'HIP10846', 'HIP11711', 'HIP11845', 'HIP13206', 'HIP13230', 'HIP13244', 'HIP13276', 'HIP13284', 'HIP13309', 'HIP13318', 'HIP13350', 'HIP13376', 'HIP13402', 'HIP13465', 'HIP13511', 'HIP13515', 'HIP13625', 'HIP13667', 'HIP13695', 'HIP13741', 'HIP13749', 'HIP13760', 'HIP13769', 'HIP13800', 'HIP13803', 'HIP13806', 'HIP13823', 'HIP13831', 'HIP13847', 'HIP13854', 'HIP13857', 'HIP13869', 'HIP13875', 'HIP13916', 'HIP13939', 'HIP13944', 'HIP13951', 'HIP13958', 'HIP13961', 'HIP13967', 'HIP13975', 'HIP13981', 'HIP13992', 'HIP14064', 'HIP14066', 'HIP14121', 'HIP14134', 'HIP14138', 'HIP14143', 'HIP14152', 'HIP14156', 'HIP14160', 'HIP14170', 'HIP14172', 'HIP14178', 'HIP14209', 'HIP14213', 'HIP14234', 'HIP14238', 'HIP14241', 'HIP14243', 'HIP14361', 'HIP14363', 'HIP14844', 'HIP15860', 'HIP16738', 'HIP17370', 'HIP17657', 'HIP17737', 'HIP17793', 'HIP17887', 'HIP19089', 'HIP19716'])
# Dir2_train = os.path.join(main_dir, 'ToyData_Train/ToyDataCleaned')
# Dir2_val = os.path.join(main_dir, 'ToyData_Val/ToyDataCleaned')

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def TokenRepToCubeVec(Dir):
    directory = os.fsencode(Dir)

    for f, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith('.xls'):
            wb = xlrd.open_workbook(Dir + '/' + filename)
            sheet = wb.sheet_by_index(0)
            arr = list(range(1, sheet.nrows))
            arr = random.sample(arr, len(arr))
            maxSeqLen = 0
            cube = np.zeros((cubeSize, 110, 4))
            for i in range(sheet.nrows - 1):

                seq = sheet.cell_value(arr[i], 1)
                if (len(seq) > maxSeqLen):
                    maxSeqLen = len(seq)

                for j, char in enumerate(seq):
                    z = i % cubeSize
                    if char == 'A':
                        cube[z, j, 0] = 1
                    elif char == 'C':
                        cube[z, j, 1] = 1
                    elif char == 'G':
                        cube[z, j, 2] = 1
                    elif char == 'T':
                        cube[z, j, 3] = 1

                if ((i + 1) % cubeSize == 0):
                    cube = cube[:, 0:maxSeqLen, :]
                    maxSeqLen = 0
                    np.save(Dir + '/cubes/' + str(f) + '_' + str(int((i) / cubeSize)) + '.npy', cube)
                    cube = np.zeros((cubeSize, 110, 4))

            # compliete last cube of rep
            for e in range((i % cubeSize) + 1, cubeSize):
                seq = sheet.cell_value(random.choice(arr), 1)
                if (len(seq) > maxSeqLen):
                    maxSeqLen = len(seq)

                for j, char in enumerate(seq):
                    z = e % cubeSize
                    if char == 'A':
                        cube[z, j, 0] = 1
                    elif char == 'C':
                        cube[z, j, 1] = 1
                    elif char == 'G':
                        cube[z, j, 2] = 1
                    elif char == 'T':
                        cube[z, j, 3] = 1

                if ((e + 1) % cubeSize == 0):
                    cube = cube[:, 0:maxSeqLen, :]
                    maxSeqLen = 0
                    np.save(Dir + '/cubes/' + str(f) + '_' + str(int((i) / cubeSize) + 1) + '.npy', cube)
                    cube = np.zeros((cubeSize, 110, 4))


def TokenRepToVecRep(Dir):
    directory = os.fsencode(Dir)

    for f, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith('.xls'):
            wb = xlrd.open_workbook(Dir + '/' + filename)
            sheet = wb.sheet_by_index(0)
            arr = list(range(sheet.nrows))
            # arr = random.sample(arr, len(arr))
            maxSeqLen = 0
            lengths = np.zeros((sheet.nrows - 1))
            Vgenes = np.zeros((sheet.nrows - 1))
            Jgenes = np.zeros((sheet.nrows - 1))
            families = np.zeros((sheet.nrows - 1))
            cube = np.zeros((sheet.nrows - 1, 120, 4))
            for i in range(sheet.nrows - 1):
                # prepare lengths
                length = sheet.cell_value(arr[i], 0)
                lengths[i] = int(length)
                # prepare V genes
                v_call = sheet.cell_value(arr[i], 2)
                start = v_call.find("-") + 1
                if (start == 0):
                    if (args.dataset == 'biomed'):
                        start = 4
                if (v_call[start + 1].isnumeric()):
                    Vgenes[i] = int(v_call[start:start + 2])
                elif (v_call[start].isnumeric()):
                    Vgenes[i] = int(v_call[start:start + 1])
                else:
                    Vgenes[i] = 0
                # prepare families
                families[i] = int(v_call[4:5])
                # prepare Jgenes
                J_call = sheet.cell_value(arr[i], 3)
                Jgenes[i] = int(J_call[4:5])
                # prepare data cubes
                seq = sheet.cell_value(arr[i], 1)
                if (length > maxSeqLen):
                    maxSeqLen = length

                for j, char in enumerate(seq):
                    if char == 'A':
                        cube[i, j, 0] = 1
                    elif char == 'C':
                        cube[i, j, 1] = 1
                    elif char == 'G':
                        cube[i, j, 2] = 1
                    elif char == 'T':
                        cube[i, j, 3] = 1

            # cube = cube[:, 0:maxSeqLen, :]
            cube = cube[:, 0:84, :]  # cut at 84 allways
            filename = filename[:-4] + '.npy'
            np.save(Dir + '/lengths/' + filename, lengths)
            np.save(Dir + '/Vgenes/' + filename, Vgenes)
            np.save(Dir + '/Jgenes/' + filename, Jgenes)
            np.save(Dir + '/families/' + filename, families)
            np.save(Dir + '/cubes/' + filename, cube)
            print(filename + '  ready')


def Remove_ShortLongSeq_mod3(Dir1, ):
    MaxSeqInLength = 84
    MinSeqInLength = 21
    directory = os.fsencode(Dir1)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.xls'):
            wb = xlrd.open_workbook(Dir1 + '/' + filename)
            sheet = wb.sheet_by_index(0)
            wb1 = Workbook()
            sheet1 = wb1.add_sheet(filename)
            removed = 0

            for i in range(sheet.nrows):
                l = sheet.cell_value(i, 0)
                if ((MaxSeqInLength >= l >= MinSeqInLength) & (l % 3 == 0)):
                    sheet1.write(i - removed, 0, l)  # l
                    sheet1.write(i - removed, 1, sheet.cell_value(i, 1))  # seq
                    sheet1.write(i - removed, 2, sheet.cell_value(i, 2))  # v
                    sheet1.write(i - removed, 3, sheet.cell_value(i, 3))  # j
                    sheet1.write(i - removed, 4, sheet.cell_value(i, 4))  # id
                else:
                    removed += 1

            print(str(removed) + " sequences removed from " + filename)
            wb1.save(Dir1 + '/Cleaned/' + filename)


"""
1. transfer data fron .tab file to .npy
2. remove short/long and %3 seq
3. prepare: junction_lengths, junctions, Vgenes, Jgenes, families  
"""


def prepare_in_one_func_biomed(Dir):
    files_counter = 0
    for file in os.listdir(Dir + '/tab'):
        filename = os.fsdecode(file)
        if filename.endswith('.tab'):
            files_counter += 1
            prepare_one_file_biomed_ray.remote(filename, Dir)
            # prepare_one_file_biomed_ray(filename, Dir)
            time.sleep(1)

    print(files_counter, ' prepared')
    time.sleep(1000)


@ray.remote
def prepare_one_file_biomed_ray(filename, Dir):
    nuc_switcher = {"A": 0, "C": 1, "G": 2, "T": 3}
    MaxSeqInLength = 96
    MinSeqInLength = 0

    csv_reader = csv.DictReader(open(Dir + '/tab/' + filename), dialect="excel-tab")
    nrows = sum(1 for row in csv_reader)
    csv_reader = csv.DictReader(open(Dir + '/tab/' + filename), dialect="excel-tab")

    junction_lengths = np.zeros((nrows - 1), dtype=np.int8)
    Vgenes = np.zeros((nrows - 1), dtype=np.int8)
    Jgenes = np.zeros((nrows - 1), dtype=np.int8)
    families = np.zeros((nrows - 1), dtype=np.int8)
    DupCount = np.zeros((nrows - 1), dtype=np.int16)
    junctions = np.zeros((nrows - 1, 120, 4), dtype=np.int8)

    removed = 0
    for i, row in enumerate(csv_reader):
        length = int(row['JUNCTION_LENGTH'])
        if ((MaxSeqInLength >= length >= MinSeqInLength) & (length % 3 == 0)):
            junc = row['JUNCTION']
            v_call = row['V_CALL']
            j_call = row['J_CALL']
            seq_id = row['SEQUENCE_ID']
            dupcount = row['DUPCOUNT']

            # prepare lengths
            junction_lengths[i - removed] = int(length)
            # prepare families
            families[i - removed] = int(v_call[4:5])
            if (v_call[5].isnumeric()):
                families[i - removed] = int(v_call[4:6])
            elif (v_call[4].isnumeric()):
                families[i - removed] = int(v_call[4:5])
            else:
                families[i - removed] = 0
                print("Family Eror in: ", filename, " value: ", v_call)
            # prepare V genes
            start = v_call.find("-") + 1
            if (start == 0):
                start = 4
            if (v_call[start + 1].isnumeric()):
                Vgenes[i - removed] = int(v_call[start:start + 2])
            elif (v_call[start].isnumeric()):
                Vgenes[i - removed] = int(v_call[start:start + 1])
            else:
                Vgenes[i - removed] = 0
                print("Vgene Eror in: ", filename, " value: ", v_call)

            # prepare Jgenes
            Jgenes[i - removed] = int(j_call[6:7])
            # prepare DupCount
            DupCount[i - removed] = int(dupcount)
            # prepare data junctions
            for j, char in enumerate(junc):
                z = nuc_switcher.get(char, 5)  # in case key is missing
                if (z == 5):
                    junctions[i - removed, :, :] = 0
                    removed += 1
                    break
                junctions[i - removed, j, z] = 1



        else:
            removed += 1

    junction_lengths = junction_lengths[0:nrows - removed]
    Vgenes = Vgenes[0:nrows - removed]
    Jgenes = Jgenes[0:nrows - removed]
    families = families[0:nrows - removed]
    DupCount = DupCount[0:nrows - removed]
    junctions = junctions[0:nrows - removed, 0:MaxSeqInLength, :]

    if (True):  # repeat by DupCount
        junction_lengths = np.repeat(junction_lengths, DupCount, axis=0)
        Vgenes = np.repeat(Vgenes, DupCount, axis=0)
        Jgenes = np.repeat(Jgenes, DupCount, axis=0)
        families = np.repeat(families, DupCount, axis=0)
        junctions = np.repeat(junctions, DupCount, axis=0)

    filename = filename[:-4] + '.npy'
    np.save(Dir + '/Cleaned_10000/lengths/' + filename, junction_lengths)
    np.save(Dir + '/Cleaned_10000/Vgenes/' + filename, Vgenes)
    np.save(Dir + '/Cleaned_10000/Jgenes/' + filename, Jgenes)
    np.save(Dir + '/Cleaned_10000/families/' + filename, families)
    np.save(Dir + '/Cleaned_10000/cubes/' + filename, junctions)
    print(filename + ': ' + str(removed) + " removed. dupcount:  " + str(len(DupCount)) + ' --> ' + str(
        len(junction_lengths)))


# def prepare_one_file_cmv_biomed(Dir):
#     MaxSeqInLength = 84
#     MinSeqInLength = 21
#     # start copy all reps
#     for file in os.listdir(Dir):
#         filename = os.fsdecode(file)
#         if filename.endswith('.tab'):
#             csv_reader = csv.DictReader(open(Dir + '/' + filename), dialect="excel-tab")
#             nrows = sum(1 for row in csv_reader)
#             csv_reader = csv.DictReader(open(Dir + '/' + filename), dialect="excel-tab")
#
#             junction_lengths = np.zeros((nrows - 1),dtype=np.int)
#             Vgenes = np.zeros((nrows - 1),dtype=np.int)
#             Jgenes = np.zeros((nrows - 1),dtype=np.int)
#             families = np.zeros((nrows - 1),dtype=np.int)
#             DupCount = np.zeros((nrows - 1),dtype=np.int)
#             junctions = np.zeros((nrows - 1, 120, 4))
#
#             removed = 0
#             for i, row in enumerate(csv_reader):
#                 length = int(row['JUNCTION_LENGTH'])
#                 if ((MaxSeqInLength >= length >= MinSeqInLength) & (length % 3 == 0)):
#                     junc = row['JUNCTION']
#                     v_call = row['V_CALL']
#                     j_call = row['J_CALL']
#                     seq_id = row['SEQUENCE_ID']
#                     dupcount = row['DUPCOUNT']
#
#                     # prepare lengths
#                     junction_lengths[i - removed] = int(length)
#                     # prepare V genes
#                     start = v_call.find("-") + 1
#                     if (start == 0):
#                         start = 4
#                     if (v_call[start + 1].isnumeric()):
#                         Vgenes[i - removed] = int(v_call[start:start + 2])
#                     elif (v_call[start].isnumeric()):
#                         Vgenes[i - removed] = int(v_call[start:start + 1])
#                     else:
#                         Vgenes[i - removed] = 0
#                         print("Vgene Eror in: ",filename," value: ", v_call)
#
#                     # prepare families
#                     families[i - removed] = int(v_call[4:5])
#                     if (v_call[5].isnumeric()):
#                         families[i - removed] = int(v_call[4:6])
#                     elif (v_call[4].isnumeric()):
#                         families[i - removed] = int(v_call[4:5])
#                     else:
#                         families[i - removed] = 0
#                         print("Family Eror in: ",filename," value: ", v_call)
#                     # prepare Jgenes
#                     Jgenes[i - removed] = int(j_call[6:7])
#                     # prepare DupCount
#                     DupCount[i - removed] = int(dupcount)
#                     # prepare data junctions
#                     for j, char in enumerate(junc):
#                         if char == 'A':
#                             junctions[i - removed, j, 0] = 1
#                         elif char == 'C':
#                             junctions[i - removed, j, 1] = 1
#                         elif char == 'G':
#                             junctions[i - removed, j, 2] = 1
#                         elif char == 'T':
#                             junctions[i - removed, j, 3] = 1
#
#
#
#                 else:
#                     removed += 1
#
#             junction_lengths = junction_lengths[0:nrows - removed]
#             Vgenes = Vgenes[0:nrows - removed]
#             Jgenes = Jgenes[0:nrows - removed]
#             families = families[0:nrows - removed]
#             DupCount = DupCount[0:nrows - removed]
#             junctions = junctions[0:nrows - removed, 0:MaxSeqInLength, :]
#
#             if(True): # repeat by DupCount
#                 junction_lengths = np.repeat(junction_lengths, DupCount, axis=0)
#                 Vgenes = np.repeat(Vgenes, DupCount, axis=0)
#                 Jgenes = np.repeat(Jgenes, DupCount, axis=0)
#                 families = np.repeat(families, DupCount, axis=0)
#                 junctions = np.repeat(junctions, DupCount, axis=0)
#
#
#             filename = filename[:-4] + '.npy'
#             np.save(Dir + '/XL/Cleaned/lengths/' + filename, junction_lengths)
#             np.save(Dir + '/XL/Cleaned/Vgenes/' + filename, Vgenes)
#             np.save(Dir + '/XL/Cleaned/Jgenes/' + filename, Jgenes)
#             np.save(Dir + '/XL/Cleaned/families/' + filename, families)
#             np.save(Dir + '/XL/Cleaned/cubes/' + filename, junctions)
#             print(filename + ': ' + str(removed) + " removed. dupcount:  " + str(len(DupCount)) + ' --> ' + str(len(junction_lengths)))


def files_names_func(data_path):
    names_list = []
    for file in os.listdir(data_path):
        if (file.endswith('.npy')):
            names_list.append(file[0:-4])
    return names_list

def prepare_in_one_func(args):
    Path(args.dest_path).mkdir(parents=True, exist_ok=True)
    Path(args.dest_path + '/lengths').mkdir(parents=True, exist_ok=True)
    Path(args.dest_path + '/Vgenes').mkdir(parents=True, exist_ok=True)
    Path(args.dest_path + '/Jgenes').mkdir(parents=True, exist_ok=True)
    Path(args.dest_path + '/families').mkdir(parents=True, exist_ok=True)
    Path(args.dest_path + '/cubes').mkdir(parents=True, exist_ok=True)
    Path(args.dest_path + '/cubes/dataINorder').mkdir(parents=True, exist_ok=True)
    Path(args.dest_path + '/cubes/familiesINorder').mkdir(parents=True, exist_ok=True)
    Path(args.dest_path + '/cubes/JgenesINorder').mkdir(parents=True, exist_ok=True)
    Path(args.dest_path + '/cubes/lengthsINorder').mkdir(parents=True, exist_ok=True)
    Path(args.dest_path + '/cubes/VgenesINorder').mkdir(parents=True, exist_ok=True)

    remove_counter = 0
    files_counter = 0

    metadata = csv.DictReader(open(args.metadata_path), delimiter='\t')
    result = []
    for r in metadata:
        filename = r["filename"]
        ill_status = r["ill_status"]
        if (ill_status == 'True'):
            tag = '_ill'
        elif (ill_status == 'False'):
            tag = ''
        else:
            print(filename + ': unknown tag')
            remove_counter += 1
            continue

        files_counter += 1
        result.append(prepare_one_file_ray_87_nucleotide.remote(filename, args.tsv_path, args.dest_path, tag,true_clusters_1))
        # prepare_one_file_ray_87_nucleotide(filename, args.tsv_path, args.dest_path, tag,true_clusters_1)
        time.sleep(args.sleep_time)
    for r in result:
        ray.wait([r])
    print(remove_counter, ' removed')
    print(files_counter, ' prepared')

def prepare_in_one_func_cmv(Dir, SampleOverview):
    remove_counter = 0
    files_counter = 0
    template_exclude =True

    for r in SampleOverview:
        filename = r["sample_name"]
        sample_tags = r["sample_tags"]
        if ((filename in exclude_list) & (template_exclude)):
            print(filename + ': excluded')
            remove_counter += 1
            continue
        elif ('Cytomegalovirus +' in sample_tags):
            tag = '_ill'
        elif ('Cytomegalovirus -' in sample_tags):
            tag = ''
        else:
            print(filename + ': unknown tag')
            remove_counter += 1
            continue

        try:
            files_counter += 1
            # prepare_one_file_cmv_ray.remote(filename, Dir, tag)
            # prepare_one_file_cmv_ray(filename, Dir, tag)
            # prepare_one_file_cmv_ray_amino_acid.remote(filename, Dir, tag)
            # prepare_one_file_cmv_ray_amino_acid(filename, Dir, tag)
            prepare_one_file_cmv_ray_87_nucleotide.remote(filename, Dir, tag)
            # prepare_one_file_cmv_ray_87_nucleotide(filename, Dir, tag)
        except:
            print("ERROR in:" + filename)
        time.sleep(0.5)

    print(remove_counter, ' removed')
    print(files_counter, ' prepared')
    time.sleep(10)


def prepare_in_one_func_covid19(Dir):
    remove_counter = 0
    files_counter = 0

    # ready_file_list = []
    # for file in os.listdir((Dir + r'/Cleaned_24000_87_median/cubes/')):
    #     ready_file_list.append(file[0:8])

    SampleOverview = csv.DictReader(open(Dir + '/' + 'all_tags_covid19.tsv', encoding="utf8"), dialect="excel-tab")

    for r in SampleOverview:
        filename = r["sample_name"]
        # if(filename in ready_file_list):
        #     print(filename + ' pass')
        #     continue
        sample_tags = r["covid_diagnosis"]
        if ('positive' in sample_tags):
            tag = '_ill'
        elif ('' in sample_tags):
            tag = ''
        else:
            print(filename + ': unknown tag')
            remove_counter += 1
            continue

        files_counter += 1
        prepare_one_file_cmv_ray_87_nucleotide.remote(filename, Dir, tag)
        # prepare_one_file_cmv_ray_87_nucleotide(filename, Dir, tag)
        time.sleep(2)

    print(remove_counter, ' removed')
    print(files_counter, ' prepared')
    time.sleep(10)


@ray.remote
def prepare_one_file_cmv_ray(filename, Dir, tag):
    gencode = {'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M', 'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
               'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K', 'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
               'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L', 'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
               'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
               'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
               'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E', 'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
               'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S', 'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
               'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_', 'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W'}
    MaxSeqInLength = 84
    MinSeqInLength = 18

    csv_reader = csv.DictReader(open(Dir + '/tsv/' + filename + '.tsv'), dialect="excel-tab")
    nrows = sum(1 for k in csv_reader)
    csv_reader = csv.DictReader(open(Dir + '/tsv/' + filename + '.tsv'), dialect="excel-tab")

    junction_lengths = np.zeros((nrows - 1), dtype=np.int)
    Vgenes = np.zeros((nrows - 1), dtype=np.int)
    Jgenes = np.zeros((nrows - 1), dtype=np.int)
    families = np.zeros((nrows - 1), dtype=np.int)
    productive_frequencies = np.zeros((nrows - 1), dtype=np.float)
    junctions = np.zeros((nrows - 1, 120, 4))

    removed = 0
    for row_number, row in enumerate(csv_reader):
        frame_type = str(row['frame_type'])
        if (frame_type == 'In'):
            # revers junction from amino acid
            rearrangement = (row['rearrangement'])
            amino_acid = (row['amino_acid'])
            seq = ''
            for frame_offset in range(3):
                for i in range(len(rearrangement) // 3):
                    seq += gencode[rearrangement[frame_offset + i * 3:frame_offset + (i + 1) * 3]]
                indx = seq.find(amino_acid)
                if (indx != -1):
                    break
            junc = rearrangement[indx * 3:(indx + len(amino_acid)) * 3]
            #
            length = int(row['cdr3_length'])
            if ((MaxSeqInLength >= length >= MinSeqInLength) & (length % 3 == 0)):
                v_gene = row['v_gene']
                v_family = row['v_family']
                j_call = row['j_gene']
                productive_frequency = row['productive_frequency']

                if ((str(v_family) == '') | (str(j_call) == '') | (str(length) == '')):
                    removed += 1
                    continue

                # prepare lengths
                junction_lengths[row_number - removed] = int(length)

                # prepare families
                if (v_family[5:7].isnumeric()):
                    families[row_number - removed] = int(v_family[5:7])
                else:
                    removed += 1
                    continue
                    # print("v_family Error in: ", filename, " value: ", v_family)

                # prepare V genes
                start = v_gene.find("-") + 1
                if (start == 0):
                    Vgenes[row_number - removed] = int(v_family[5:7])
                elif (v_gene[start + 1].isnumeric()):
                    Vgenes[row_number - removed] = int(v_gene[start:start + 2])
                elif (v_gene[start].isnumeric()):
                    Vgenes[row_number - removed] = int(v_gene[start:start + 1])
                else:
                    Vgenes[row_number - removed] = families[row_number - removed]
                    # print("Vgene Error in: ", filename, " value: ", v_gene)

                # prepare Jgenes
                if (j_call[9:10].isnumeric()):
                    Jgenes[row_number - removed] = int(j_call[9:10])
                else:
                    removed += 1
                    continue
                    # print("Jgene Error in: ", filename, " value: ", j_call)
                # prepare productive_frequencies
                # print(float(productive_frequency))
                productive_frequencies[row_number - removed] = float(productive_frequency)
                # if(productive_frequency.isnumeric()):
                #     productive_frequency[row_number - removed] = abs(int(productive_frequency))
                # else:
                #     productive_frequency[row_number - removed] = 0.00000000001

                # prepare data junctions
                for j, char in enumerate(junc):
                    if char == 'A':
                        junctions[row_number - removed, j, 0] = 1
                    elif char == 'C':
                        junctions[row_number - removed, j, 1] = 1
                    elif char == 'G':
                        junctions[row_number - removed, j, 2] = 1
                    elif char == 'T':
                        junctions[row_number - removed, j, 3] = 1



            else:
                removed += 1

        else:
            removed += 1

    junction_lengths = junction_lengths[0:nrows - removed]
    Vgenes = Vgenes[0:nrows - removed]
    Jgenes = Jgenes[0:nrows - removed]
    families = families[0:nrows - removed]
    productive_frequencies = productive_frequencies[0:nrows - removed]
    junctions = junctions[0:nrows - removed, 0:MaxSeqInLength, :]

    if (True):  # repeat by productive_frequencies
        productive_frequencies = np.floor(productive_frequencies * nrows) + 1
        productive_frequencies = productive_frequencies.astype(int)
        junction_lengths = np.repeat(junction_lengths, productive_frequencies, axis=0)
        Vgenes = np.repeat(Vgenes, productive_frequencies, axis=0)
        Jgenes = np.repeat(Jgenes, productive_frequencies, axis=0)
        families = np.repeat(families, productive_frequencies, axis=0)
        junctions = np.repeat(junctions, productive_frequencies, axis=0)

    filename = filename + tag + '.npy'
    np.save(Dir + '/Cleaned_22500/lengths/' + filename, junction_lengths)
    np.save(Dir + '/Cleaned_22500/Vgenes/' + filename, Vgenes)
    np.save(Dir + '/Cleaned_22500/Jgenes/' + filename, Jgenes)
    np.save(Dir + '/Cleaned_22500/families/' + filename, families)
    np.save(Dir + '/Cleaned_22500/cubes/' + filename, junctions)
    print(filename + ': ' + str(removed) + " removed. dupcount:  " + str(len(productive_frequencies)) + ' --> ' + str(
        len(junction_lengths)))
    print("%d bytes" % (junctions.size * junctions.itemsize))
    data = np.load(Dir + '/Cleaned_22500/cubes/' + filename)
    print("%d bytes data" % (data.size * data.itemsize))


@ray.remote
def prepare_one_file_cmv_ray_amino_acid(filename, Dir, tag):
    AA_switcher = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "E": 5, "Q": 6, "G": 7, "H": 8, "I": 9, "L": 10, "K": 11,
                   "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19}
    csv_reader = csv.DictReader(open(Dir + '/tsv/' + filename + '.tsv'), dialect="excel-tab")
    nrows = sum(1 for k in csv_reader)
    csv_reader = csv.DictReader(open(Dir + '/tsv/' + filename + '.tsv'), dialect="excel-tab")

    junction_lengths = np.zeros((nrows - 1), dtype=np.int8)
    Vgenes = np.zeros((nrows - 1), dtype=np.int8)
    Jgenes = np.zeros((nrows - 1), dtype=np.int8)
    families = np.zeros((nrows - 1), dtype=np.int8)
    productive_frequencies = np.zeros((nrows - 1), dtype=np.float)
    junctions = np.zeros((nrows - 1, 27, 20), dtype=np.int8)

    removed = 0
    for row_number, row in enumerate(csv_reader):
        frame_type = str(row['frame_type'])
        if (frame_type == 'In'):
            amino_acid = (row['amino_acid'])
            length = int(row['cdr3_length'])
            v_gene = row['v_gene']
            v_family = row['v_family']
            j_call = row['j_gene']
            productive_frequency = row['productive_frequency']

            if ((str(v_family) == '') | (str(j_call) == '') | (str(length) == '')):
                removed += 1
                continue

            # prepare lengths
            junction_lengths[row_number - removed] = int(length)

            # prepare families
            if (v_family[5:7].isnumeric()):
                families[row_number - removed] = int(v_family[5:7])
            else:
                removed += 1
                continue

            # prepare V genes
            start = v_gene.find("-") + 1
            if (start == 0):
                Vgenes[row_number - removed] = int(v_family[5:7])
            elif (v_gene[start + 1].isnumeric()):
                Vgenes[row_number - removed] = int(v_gene[start:start + 2])
            elif (v_gene[start].isnumeric()):
                Vgenes[row_number - removed] = int(v_gene[start:start + 1])
            else:
                Vgenes[row_number - removed] = families[row_number - removed]

            # prepare Jgenes
            if (j_call[9:10].isnumeric()):
                Jgenes[row_number - removed] = int(j_call[9:10])
            else:
                removed += 1
                continue

            productive_frequencies[row_number - removed] = float(productive_frequency)

            for j, char in enumerate(amino_acid):
                junctions[row_number - removed, j, AA_switcher[char]] = 1

        else:
            removed += 1

    junction_lengths = junction_lengths[0:nrows - removed]
    Vgenes = Vgenes[0:nrows - removed]
    Jgenes = Jgenes[0:nrows - removed]
    families = families[0:nrows - removed]
    productive_frequencies = productive_frequencies[0:nrows - removed]
    junctions = junctions[0:nrows - removed, :, :]

    if (True):  # repeat by productive_frequencies
        productive_frequencies = np.floor(productive_frequencies * nrows) + 1
        productive_frequencies = productive_frequencies.astype(int)
        junction_lengths = np.repeat(junction_lengths, productive_frequencies, axis=0)
        Vgenes = np.repeat(Vgenes, productive_frequencies, axis=0)
        Jgenes = np.repeat(Jgenes, productive_frequencies, axis=0)
        families = np.repeat(families, productive_frequencies, axis=0)
        junctions = np.repeat(junctions, productive_frequencies, axis=0)

    filename = filename + tag + '.npy'
    np.save(Dir + '/Cleaned_32767/lengths/' + filename, junction_lengths)
    np.save(Dir + '/Cleaned_32767/Vgenes/' + filename, Vgenes)
    np.save(Dir + '/Cleaned_32767/Jgenes/' + filename, Jgenes)
    np.save(Dir + '/Cleaned_32767/families/' + filename, families)
    np.save(Dir + '/Cleaned_32767/cubes/' + filename, junctions)
    print(filename + ': ' + str(removed) + " removed. dupcount:  " + str(len(productive_frequencies)) + ' --> ' + str(
        len(junction_lengths)))
    # print("%d bytes" % (junctions.size * junctions.itemsize))

@ray.remote
def prepare_one_file_ray_87_nucleotide(filename, tsv_path, dest_path, tag,true_clusters_1):
    minLen = 27
    maxLen = 81
    nuc_switcher = {"A": 0, "C": 1, "G": 2, "T": 3}
    csv_reader = csv.DictReader(open(tsv_path + filename), dialect="excel-tab")
    nrows = sum(1 for k in csv_reader)
    csv_reader = csv.DictReader(open(tsv_path + filename), dialect="excel-tab")

    lengths = np.zeros((nrows), dtype=np.int8)
    Vgenes = np.zeros((nrows), dtype=np.int8)
    Jgenes = np.zeros((nrows), dtype=np.int8)
    families = np.zeros((nrows), dtype=np.int8)
    allTemplate = np.zeros((nrows), dtype=np.int64)
    nuc_rearrangement = np.zeros((nrows, 87, 4), dtype=np.int8)

    removed = 0
    for row_number, row in enumerate(csv_reader):
        rearrangement = (row['rearrangement'])
        length = int(row['cdr3_length'])
        v_gene = row['v_gene']
        v_family = row['v_family']
        j_call = row['j_gene']

        templates = row['templates']

        # if ((str(v_family) == '') | (str(j_call) == '') | (str(length) == '')):
        #     removed += 1
        #     continue
        if (str(length) == ''):
            length = '0'

        if not (minLen <= length <= maxLen):
            removed += 1
            continue
        if (int(templates)>70):
            removed += 1
            continue

        if (v_family[5:7].isnumeric()):
            f = str(int(v_family[5:7]))
        else:
            f = '0'
            # removed += 1
            # continue

        # prepare lengths
        l = str(length)
        # prepare V genes
        start = v_gene.find("-") + 1
        if (start == 0):
            v = '0'
            # Vgenes[row_number - removed] = int(v_family[5:7])
        elif (v_gene[start + 1].isnumeric()):
            v = str(int(v_gene[start:start + 2]))
        elif (int(v_gene[start].isnumeric())):
            v = str(v_gene[start:start + 1])
        else:
            v = '0'
            # Vgenes[row_number - removed] = families[row_number - removed]

        # prepare Jgenes
        if (j_call[9:10].isnumeric()):
            j = str(j_call[9:10])
        else:
            j = '0'
            # removed += 1
            # continue

        key = j + '_' + f + '_' + v + '_' + l

        # if (all_j_f_v_l_percent[key] < 0.000001):
        #     removed += 1
        #     continue
        # if (key in test_list):
        #     # if (true_clusters_1[key]<=1):
        #     # if (true_dic[key]<=1) & (false_dic[key]<=1):
        #     if (test_list[key]<=2):
        #         removed += 1
        #         continue
        # else:
        #     removed += 1
        #     continue

        Jgenes[row_number - removed], families[row_number - removed], Vgenes[row_number - removed], lengths[row_number - removed] = int(j), int(f), int(v), int(l)
        allTemplate[row_number - removed] = max(1,int(templates)) #the max for HIP04958 HIP14092 HIP14106 that somtimes have -1

        for j, char in enumerate(rearrangement):
            if char == 'N':
                continue
            nuc_rearrangement[row_number - removed, j, nuc_switcher[char]] = 1


    lengths = lengths[0:nrows - removed]
    Vgenes = Vgenes[0:nrows - removed]
    Jgenes = Jgenes[0:nrows - removed]
    families = families[0:nrows - removed]
    allTemplate = allTemplate[0:nrows - removed]
    nuc_rearrangement = nuc_rearrangement[0:nrows - removed, :, :]

    repeat_by_allTemplate = True

    if (repeat_by_allTemplate):  # repeat by allTemplate
        lengths = np.repeat(lengths, allTemplate, axis=0)
        Vgenes = np.repeat(Vgenes, allTemplate, axis=0)
        Jgenes = np.repeat(Jgenes, allTemplate, axis=0)
        families = np.repeat(families, allTemplate, axis=0)
        nuc_rearrangement = np.repeat(nuc_rearrangement, allTemplate, axis=0)

    filename = filename[0:8] + tag + '.npy'
    np.save(dest_path + '/lengths/' + filename, lengths)
    np.save(dest_path + '/Vgenes/' + filename, Vgenes)
    np.save(dest_path + '/Jgenes/' + filename, Jgenes)
    np.save(dest_path + '/families/' + filename, families)
    np.save(dest_path + '/cubes/' + filename, nuc_rearrangement)
    print(filename + ': ' + str(removed) + " removed. dupcount:  " + str(len(allTemplate)) + ' --> ' + str(
        len(lengths)))
    # print("%d bytes" % (junctions.size * junctions.itemsize))

    return True

# backup before unine under prepare_in_one_func (when i worked on simulation)
@ray.remote
def prepare_one_file_cmv_ray_87_nucleotide(filename, Dir, tag):
    minLen = 9
    maxLen = 75
    nuc_switcher = {"A": 0, "C": 1, "G": 2, "T": 3}
    csv_reader = csv.DictReader(open(Dir + '/tsv/' + filename + '.tsv'), dialect="excel-tab")
    nrows = sum(1 for k in csv_reader)
    csv_reader = csv.DictReader(open(Dir + '/tsv/' + filename + '.tsv'), dialect="excel-tab")

    junction_lengths = np.zeros((nrows - 1), dtype=np.int8)
    Vgenes = np.zeros((nrows - 1), dtype=np.int8)
    Jgenes = np.zeros((nrows - 1), dtype=np.int8)
    families = np.zeros((nrows - 1), dtype=np.int8)
    allTemplate = np.zeros((nrows - 1), dtype=np.int64)
    # frequencies = np.zeros((nrows - 1), dtype=np.float64)
    nuc_rearrangement = np.zeros((nrows - 1, 87, 4), dtype=np.int8)

    removed = 0
    flag = True
    for row_number, row in enumerate(csv_reader):
        rearrangement = (row['rearrangement'])
        length = int(row['cdr3_length'])
        v_gene = row['v_gene']
        v_family = row['v_family']
        j_call = row['j_gene']
        # frequency = row['productive_frequency']  # covid19
        # frequency = row['frequency'] # cmv
        templates = row['templates']

        if ((str(v_family) != '') & (str(j_call) != '') & (str(length) != '')):
            # prepare lengths
            junction_lengths[row_number - removed] = length
            if not (minLen <= length <= maxLen):
                removed += 1
                continue

            # if (isfloat(frequency)):
            #     frequencies[row_number - removed] = float(frequency)
            # else:
            #     removed += 1
            #     continue

            # prepare families
            if (v_family[5:7].isnumeric()):
                families[row_number - removed] = int(v_family[5:7])
            else:
                removed += 1
                continue

            # prepare V genes
            start = v_gene.find("-") + 1
            if (start == 0):
                Vgenes[row_number - removed] = int(v_family[5:7])
            elif (v_gene[start + 1].isnumeric()):
                Vgenes[row_number - removed] = int(v_gene[start:start + 2])
            elif (v_gene[start].isnumeric()):
                Vgenes[row_number - removed] = int(v_gene[start:start + 1])
            else:
                Vgenes[row_number - removed] = families[row_number - removed]

            # prepare Jgenes
            if (j_call[9:10].isnumeric()):
                Jgenes[row_number - removed] = int(j_call[9:10])
            else:
                removed += 1
                continue

            allTemplate[row_number - removed] = max(1,int(templates)) #the max for HIP04958 HIP14092 HIP14106 that somtimes have -1

            for j, char in enumerate(rearrangement):
                if char == 'N':
                    continue
                nuc_rearrangement[row_number - removed, j, nuc_switcher[char]] = 1

        else:
            removed += 1

    junction_lengths = junction_lengths[0:nrows - removed]
    Vgenes = Vgenes[0:nrows - removed]
    Jgenes = Jgenes[0:nrows - removed]
    families = families[0:nrows - removed]
    # frequencies = frequencies[0:nrows - removed]
    allTemplate = allTemplate[0:nrows - removed]
    nuc_rearrangement = nuc_rearrangement[0:nrows - removed, :, :]

    repeat_by_allTemplate = True
    repeat_by_frequencies = True
    remove_frequencies = False
    clip_frequencies = False

    if (repeat_by_allTemplate):  # repeat by allTemplate
        junction_lengths = np.repeat(junction_lengths, allTemplate, axis=0)
        Vgenes = np.repeat(Vgenes, allTemplate, axis=0)
        Jgenes = np.repeat(Jgenes, allTemplate, axis=0)
        families = np.repeat(families, allTemplate, axis=0)
        nuc_rearrangement = np.repeat(nuc_rearrangement, allTemplate, axis=0)

    # if (repeat_by_frequencies):  # repeat by frequencies
    #     if (remove_frequencies):  # remove frequencies below median
    #         median = np.percentile(frequencies, 30)
    #         frequencies[frequencies < median] = 0
    #
    #     total_reads = 1 / np.min(frequencies)
    #     frequencies = np.ceil((frequencies * total_reads))
    #     if (clip_frequencies):  # clip frequencies
    #         frequencies = np.clip(frequencies, 0, 1000)
    #
    #     frequencies = frequencies.astype(int)
    #     junction_lengths = np.repeat(junction_lengths, frequencies, axis=0)
    #     Vgenes = np.repeat(Vgenes, frequencies, axis=0)
    #     Jgenes = np.repeat(Jgenes, frequencies, axis=0)
    #     families = np.repeat(families, frequencies, axis=0)
    #     nuc_rearrangement = np.repeat(nuc_rearrangement, frequencies, axis=0)

    filename = filename + tag + '.npy'
    np.save(Dir + '/Cleaned_32000_87_templates_folds/lengths/' + filename, junction_lengths)
    np.save(Dir + '/Cleaned_32000_87_templates_folds/Vgenes/' + filename, Vgenes)
    np.save(Dir + '/Cleaned_32000_87_templates_folds/Jgenes/' + filename, Jgenes)
    np.save(Dir + '/Cleaned_32000_87_templates_folds/families/' + filename, families)
    np.save(Dir + '/Cleaned_32000_87_templates_folds/cubes/' + filename, nuc_rearrangement)
    print(filename + ': ' + str(removed) + " removed. dupcount:  " + str(len(allTemplate)) + ' --> ' + str(
        len(junction_lengths)))
    # print("%d bytes" % (junctions.size * junctions.itemsize))


default_data_path_celiac = r'C:\Users\user\Desktop\limudim\M.Sc\RepertoiresClassification\randomNsec\Data\celiac_data\XL'
default_data_path_biomed = r'C:\Users\user\Desktop\limudim\M.Sc\RepertoiresClassification\datasets\BIOMED2'
default_data_path_cmv = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV'
default_metadata_path = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/simulated_data/metadata.csv'
default_tsv_path = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/simulated_data1/'
default_dest_path = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/Cleaned'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset', help='biomed or celiac', type=str, default='biomed')
    parser.add_argument('--mode', help='TokenRepToCubeVec(Dir2_train) / TokenRepToVecRep(Dir2_train)', type=str, default='prepare_in_one_func')
    parser.add_argument('--data_path', help='data_path', type=str, default=default_data_path_cmv)
    parser.add_argument('--tsv_path', help='tsv_path', type=str, default=default_tsv_path)
    parser.add_argument('--dest_path', help='dest_path', type=str, default=default_dest_path)
    parser.add_argument('--sleep_time', type=float, default=1)
    parser.add_argument('--GIB_per_obj', type=int, default=1)
    parser.add_argument('--ray_cpus', type=int, default=48)
    args = parser.parse_args()
    args.metadata_path = args.tsv_path + 'metadata.csv'

    ray.init(object_store_memory=args.GIB_per_obj * 1024 * 1024 * 1024, num_cpus=args.ray_cpus)

    if args.mode == 'TokenRepToCubeVec':
        print('TokenRepToCubeVec')
        TokenRepToCubeVec(args.data_path)
    elif args.mode == 'TokenRepToVecRep':
        print('TokenRepToVecRep')
        TokenRepToVecRep(args.data_path + '/Cleaned')
    elif args.mode == 'Remove_ShortLongSeq_mod3':
        print('Remove_ShortLongSeq_mod3')
        Remove_ShortLongSeq_mod3(args.data_path)
    elif args.mode == 'prepare_in_one_func_biomed':
        print('prepare_in_one_func_biomed')
        prepare_in_one_func_biomed(args.data_path)
    elif args.mode == 'prepare_in_one_func_cmv':
        print('prepare_in_one_func_cmv')
        SampleOverview = csv.DictReader(open(args.data_path + '/' + 'cohort1.tsv'), dialect="excel-tab")
        prepare_in_one_func_cmv(args.data_path, SampleOverview)
        SampleOverview = csv.DictReader(open(args.data_path + '/' + 'cohort2.tsv'), dialect="excel-tab")
        prepare_in_one_func_cmv(args.data_path, SampleOverview)
    elif args.mode == 'prepare_in_one_func_covid19':
        print('prepare_in_one_func_covid19')
        prepare_in_one_func_covid19(args.data_path)
    elif args.mode == 'prepare_in_one_func':
        print('prepare_in_one_func')
        prepare_in_one_func(args)

    print('end')
