import os
import random
from pathlib import Path
import numpy as np
import argparse
import csv
import time
import ray

ray.init()

Known_exclude_list = list(['HIP00614', 'HIP00710', 'HIP00779', 'HIP00951', 'HIP01181', 'HIP01298', 'HIP01359', 'HIP01501', 'HIP01765', 'HIP01867', 'HIP03385', 'HIP03484', 'HIP03685', 'HIP04455', 'HIP04464', 'HIP05552', 'HIP05757', 'HIP05934', 'HIP05941', 'HIP05948', 'HIP08339', 'HIP08596', 'HIP08821', 'HIP09681', 'HIP10424', 'HIP10815', 'HIP10846', 'HIP11711', 'HIP11845', 'HIP13206', 'HIP13230', 'HIP13244', 'HIP13276', 'HIP13284', 'HIP13309', 'HIP13318', 'HIP13350', 'HIP13376', 'HIP13402', 'HIP13465', 'HIP13511', 'HIP13515', 'HIP13625', 'HIP13667', 'HIP13695', 'HIP13741', 'HIP13749', 'HIP13760', 'HIP13769', 'HIP13800', 'HIP13803', 'HIP13806', 'HIP13823', 'HIP13831', 'HIP13847', 'HIP13854', 'HIP13857', 'HIP13869', 'HIP13875', 'HIP13916', 'HIP13939', 'HIP13944', 'HIP13951', 'HIP13958', 'HIP13961', 'HIP13967', 'HIP13975', 'HIP13981', 'HIP13992', 'HIP14064', 'HIP14066', 'HIP14121', 'HIP14134', 'HIP14138', 'HIP14143', 'HIP14152', 'HIP14156', 'HIP14160', 'HIP14170', 'HIP14172', 'HIP14178', 'HIP14209', 'HIP14213', 'HIP14234', 'HIP14238', 'HIP14241', 'HIP14243', 'HIP14361', 'HIP14363', 'HIP14844', 'HIP15860', 'HIP16738', 'HIP17370', 'HIP17657', 'HIP17737', 'HIP17793', 'HIP17887', 'HIP19089', 'HIP19716'])
known_l_f_j_v_list = list(['42_6_2_5', '45_6_2_5', '42_5_1_1', '42_5_2_1', '39_5_1_1', '39_6_1_5', '42_5_5_1', '45_2_5_1', '42_7_1_9', '45_9_1_1', '39_7_1_9', '45_6_1_5', '45_5_1_1', '42_6_1_5', '45_6_3_5', '42_6_5_5'])
known_l_f_j_v_rate_list = list(['0.001800', '0.001712', '0.003431', '0.002106', '0.002036', '0.001854', '0.002037', '0.001482', '0.001970', '0.002324', '0.001584', '0.002357', '0.003691', '0.002250', '0.001218', '0.001463'])
gencode = {'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M', 'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
           'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K', 'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
           'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L', 'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
           'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
           'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
           'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E', 'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
           'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S', 'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
           'TAC': 'Y', 'TAT': 'Y', 'TAA': 'K', 'TAG': 'P', 'TGC': 'C', 'TGT': 'C', 'TGA': 'L', 'TGG': 'W'}
           # 'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_', 'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W'} real, but all 3 '_' are stop codon

class file_info(object):
    def __init__(self, filename,file_size):
        self.filename = filename
        self.j_f_v_l_dic = {}
        self.file_size = file_size
        self.rearrangement_count = 0
        self.total_rearrangement_count = 0
        self.signal_count = 0
        self.total_signal_count = 0
        self.cmv = False
        self.ill_status = False
        self.j_f_v_l_list = set([])
        self.file_len = 0


    def get_max_element(self):
        return sorted(self.j_f_v_l_dic, key=self.j_f_v_l_dic.get, reverse=True)[:300]

def analyze_clones_appearance_and_average_sizes(args):
    file_list = []
    # cou = 0
    for f in os.listdir(args.cmv_path):
        # cou += 1
        filename = os.fsdecode(f)
        if ((filename.endswith('.tsv')) and (filename not in Known_exclude_list)):
            file_size = int((os.stat(args.cmv_path + filename).st_size -128) / 8)
            file_class = file_info(filename,file_size)
            with open(args.cmv_path + filename) as file:
                csv_reader = csv.DictReader(file, dialect="excel-tab")
                for row_number, row in enumerate(csv_reader):
                    if row['frame_type'] == 'In':
                        l = int(row['cdr3_length'])
                        v_gene = row['v_gene']
                        v_family = row['v_family']
                        j_call = row['j_gene']

                        if ((str(v_family) != '') & (str(j_call) != '') & (str(l) != '')):

                            # prepare families
                            if (v_family[5:7].isnumeric()):
                                if (v_family[5:6] == '0'):
                                    f = v_family[6:7]
                                else:
                                    f = v_family[5:7]

                            else:
                                continue

                            # prepare V genes
                            start = v_gene.find("-") + 1
                            if (start == 0):
                                continue
                            elif (v_gene[start + 1].isnumeric()):
                                if (v_gene[start:start + 1] == '0'):
                                    v = v_gene[start + 1:start + 2]
                                else:
                                    v = v_gene[start:start + 2]
                            elif (v_gene[start].isnumeric()):
                                v = v_gene[start:start + 1]
                            else:
                                continue
                            # prepare Jgenes
                            if (j_call[9:10].isnumeric()):
                                j = int(j_call[9:10])
                            else:
                                continue

                            key = str(j)+'_'+str(f)+'_'+str(v)+'_'+str(l)
                            file_class.j_f_v_l_list.add(key)
                            if key not in file_class.j_f_v_l_dic:
                                file_class.j_f_v_l_dic[key] = 1
                            else:
                                file_class.j_f_v_l_dic[key] += 1

                file_class.file_len = row_number
                file_list.append(file_class)
                print(filename)
                # if cou == 5:
                #     break


    # get all j_f_v_l
    all_j_f_v_l_list = set([])
    for f in file_list:
        all_j_f_v_l_list = set.union(all_j_f_v_l_list, f.j_f_v_l_list)

    # get all j_f_v_l clones appearance count in repertoires
    j_f_v_l_clones_count_dic = {key: 0 for key in all_j_f_v_l_list}
    for f in file_list:
        for key in f.j_f_v_l_list:
            j_f_v_l_clones_count_dic[key] += 1

    # remove clones below thresholds appearance
    j_f_v_l_clones_count_dic = {key: j_f_v_l_clones_count_dic[key]
        for key in j_f_v_l_clones_count_dic if j_f_v_l_clones_count_dic[key] >= args.threshold}

    # get clones average sizes
    average_sizes_dic = {key: 0 for key in j_f_v_l_clones_count_dic}
    for f in file_list:
        for key in average_sizes_dic:
            if key in f.j_f_v_l_dic:
                average_sizes_dic[key] += f.j_f_v_l_dic[key]/f.file_len
    average_sizes_dic = { key: average_sizes_dic[key]/j_f_v_l_clones_count_dic[key] for key in average_sizes_dic} # normalize

    np.save('average_sizes_dic.npy', average_sizes_dic)
    print(average_sizes_dic)
    # Load
    read_dictionary = np.load('average_sizes_dic.npy',allow_pickle='TRUE').item()
    print(read_dictionary)

# need to change l_f_j_l to j_f_v_l
def analyze(args):
    file_list = []
    for f in os.listdir(args.cmv_path):
        filename = os.fsdecode(f)
        if ((filename.endswith('.tsv')) and (filename not in Known_exclude_list)):
            file_size = int((os.stat(args.cmv_path + filename).st_size -128) / 8)
            file_class = file_info(filename,file_size)
            with open(args.cmv_path + filename) as file:
                csv_reader = csv.DictReader(file, dialect="excel-tab")
                for row_number, row in enumerate(csv_reader):
                    if row['frame_type'] == 'In':
                        l = int(row['cdr3_length'])
                        v_gene = row['v_gene']
                        v_family = row['v_family']
                        j_call = row['j_gene']

                        if ((str(v_family) != '') & (str(j_call) != '') & (str(l) != '')):

                            # prepare families
                            if (v_family[5:7].isnumeric()):
                                if (v_family[5:6] == '0'):
                                    f = v_family[6:7]
                                else:
                                    f = v_family[5:7]

                            else:
                                continue

                            # prepare V genes
                            start = v_gene.find("-") + 1
                            if (start == 0):
                                continue
                            elif (v_gene[start + 1].isnumeric()):
                                if (v_gene[start:start + 1] == '0'):
                                    v = v_gene[start + 1:start + 2]
                                else:
                                    v = v_gene[start:start + 2]
                            elif (v_gene[start].isnumeric()):
                                v = v_gene[start:start + 1]
                            else:
                                continue
                            # prepare Jgenes
                            if (j_call[9:10].isnumeric()):
                                j = int(j_call[9:10])
                            else:
                                continue

                            key = str(l)+'_'+str(f)+'_'+str(j)+'_'+str(v)
                            file_class.l_f_j_v_list.add(key)
                            if key not in file_class.l_f_j_v_dic:
                                file_class.l_f_j_v_dic[key] = 1
                            else:
                                file_class.l_f_j_v_dic[key] += 1

                file_class.file_len = row_number
                file_list.append(file_class)
                print(filename)

    max_trio = set(file_list[0].get_max_element())
    for f in file_list:
        max_trio.intersection_update(f.get_max_element())
    print(max_trio)

    ave_trio = np.zeros(len(max_trio))
    for f in file_list:
        for j, trio in enumerate(max_trio):
            ave_trio[j] += f.l_f_j_v_dic[trio] / f.file_len

    ave_trio /= len(file_list)
    # print(ave_trio)
    print(["{0:0.6f}".format(i) for i in ave_trio])

    # l_f_j_v analysis
    all_l_f_j_v_list = set([])
    common_l_f_j_v_list = set([])
    for f in file_list:
        all_l_f_j_v_list = set.union(all_l_f_j_v_list, f.l_f_j_v_list)

    for key in file_list[0].l_f_j_v_list:
        flag = True
        for f in file_list:
            if key not in f.l_f_j_v_list:
                flag = False
                break
        if flag:
            common_l_f_j_v_list.add(key)
    all_l_f_j_v_percent = {}
    for f in file_list:
        for key in f.l_f_j_v_dic:
            if key not in all_l_f_j_v_percent:
                all_l_f_j_v_percent[key] = f.l_f_j_v_dic[key]/f.file_len
            else:
                all_l_f_j_v_percent[key] += f.l_f_j_v_dic[key]/f.file_len
    for key in all_l_f_j_v_percent:
        all_l_f_j_v_percent[key] = all_l_f_j_v_percent[key]/len(file_list)

    print('all_l_f_j_v_list:')
    print(len(all_l_f_j_v_list))
    print(all_l_f_j_v_list)

    print('common_l_f_j_v_list:')
    print(len(common_l_f_j_v_list))
    print(common_l_f_j_v_list)

    print('all_l_f_j_v_percent:')
    print(len(all_l_f_j_v_percent))
    print(all_l_f_j_v_percent)


    # with open('all_l_f_j_v_list.txt', 'w') as f:
    #     for item in l_f_j_v_list:
    #         f.write("%s\n" % item)


def implementSignal(args, rearrangement, amino_acid):
    if random.uniform(0, 1) <= args.implementing_rate:
        seq = ''
        for frame_offset in range(3):
            for i in range(len(rearrangement) // 3):
                seq += gencode[rearrangement[frame_offset + i * 3:frame_offset + (i + 1) * 3]]
            cdr3_start = seq.find(amino_acid)
            if (cdr3_start != -1):
                break

        position = 3 * random.randint(0,args.position_range)

        new_signal = ''
        for i in range(len(args.signal)):
            if random.uniform(0, 1) < args.HD_probabilities:
                new_signal = new_signal + args.signal[i]
            else:
                new_signal = new_signal + rearrangement[cdr3_start + position + i]

        new_rearrangement = rearrangement[0:cdr3_start + position] + new_signal + rearrangement[cdr3_start + position + len(args.signal):]
        new_AA_signal = ''
        for i in range(len(new_signal)//3):
            new_AA_signal = new_AA_signal + gencode[new_signal[i*3:(i+1)*3]]
        new_amino_acid = amino_acid[0:(cdr3_start + position)//3] + new_AA_signal + amino_acid[(cdr3_start + position + len(args.signal))//3:]

        return True, new_rearrangement, new_amino_acid

    else:
        return False, rearrangement, amino_acid

@ray.remote
def simulate_one_file(args, filename, ill_flag):
    fieldnames = ['rearrangement', 'amino_acid', 'frame_type', 'templates', 'cdr3_length', 'v_family', 'v_gene','v_allele', 'j_family', 'j_gene', 'j_allele', 'locus', 'tags']
    with open(args.cmv_path + filename) as old_file:
        old_csv_reader = csv.DictReader(old_file, dialect="excel-tab")
        with open(args.tsv_path + filename, 'w') as new_file:
            new_csv_reader = csv.DictWriter(new_file, fieldnames=fieldnames, delimiter='\t')
            # new_csv_reader.writerow({'rearrangement': 'rearrangement', 'amino_acid': 'amino_acid', 'frame_type': 'frame_type', 'rearrangement_type': 'rearrangement_type', 'templates': 'templates', 'reads': 'reads', 'frequency': 'frequency', 'productive_frequency': 'productive_frequency', 'cdr3_length': 'cdr3_length', 'v_family': 'v_family', 'v_gene': 'v_gene', 'v_allele': 'v_allele', 'd_family': 'd_family', 'd_gene': 'd_gene', 'd_allele': 'd_allele', 'j_family': 'j_family', 'j_gene': 'j_gene', 'j_allele': 'j_allele', 'v_deletions': 'v_deletions', 'd5_deletions': 'd5_deletions', 'd3_deletions': 'd3_deletions', 'j_deletions': 'j_deletions', 'n2_insertions': 'n2_insertions', 'n1_insertions': 'n1_insertions', 'v_index': 'v_index', 'n1_index': 'n1_index', 'n2_index': 'n2_index', 'd_index': 'd_index', 'j_index': 'j_index', 'v_family_ties': 'v_family_ties', 'v_gene_ties': 'v_gene_ties', 'v_allele_ties': 'v_allele_ties', 'd_family_ties': 'd_family_ties', 'd_gene_ties': 'd_gene_ties', 'd_allele_ties': 'd_allele_ties', 'j_family_ties': 'j_family_ties', 'j_gene_ties': 'j_gene_ties', 'j_allele_ties': 'j_allele_ties', 'sequence_tags': 'sequence_tags', 'v_shm_count': 'v_shm_count', 'v_shm_indexes': 'v_shm_indexes', 'antibody': 'antibody', 'sample_name': 'sample_name', 'species': 'species', 'locus': 'locus', 'product_subtype': 'product_subtype', 'kit_pool': 'kit_pool', 'total_templates': 'total_templates', 'productive_templates': 'productive_templates', 'outofframe_templates': 'outofframe_templates', 'stop_templates': 'stop_templates', 'dj_templates': 'dj_templates', 'total_rearrangements': 'total_rearrangements', 'productive_rearrangements': 'productive_rearrangements', 'outofframe_rearrangements': 'outofframe_rearrangements', 'stop_rearrangements': 'stop_rearrangements', 'dj_rearrangements': 'dj_rearrangements', 'total_reads': 'total_reads', 'total_productive_reads': 'total_productive_reads', 'total_outofframe_reads': 'total_outofframe_reads', 'total_stop_reads': 'total_stop_reads', 'total_dj_reads': 'total_dj_reads', 'productive_clonality': 'productive_clonality', 'productive_entropy': 'productive_entropy', 'sample_clonality': 'sample_clonality', 'sample_entropy': 'sample_entropy', 'sample_amount_ng': 'sample_amount_ng', 'sample_cells_mass_estimate': 'sample_cells_mass_estimate', 'fraction_productive_of_cells_mass_estimate': 'fraction_productive_of_cells_mass_estimate', 'sample_cells': 'sample_cells', 'fraction_productive_of_cells': 'fraction_productive_of_cells', 'max_productive_frequency': 'max_productive_frequency', 'max_frequency': 'max_frequency', 'counting_method': 'counting_method', 'primer_set': 'primer_set', 'release_date': 'release_date', 'sample_tags': 'sample_tags', 'fraction_productive': 'fraction_productive', 'order_name': 'order_name', 'kit_id': 'kit_id', 'total_t_cells': 'total_t_cells'}
            new_csv_reader.writerow(
                {'rearrangement': 'rearrangement', 'amino_acid': 'amino_acid', 'frame_type': 'frame_type',
                 'templates': 'templates', 'cdr3_length': 'cdr3_length', 'v_family': 'v_family', 'v_gene': 'v_gene',
                 'v_allele': 'v_allele', 'j_family': 'j_family', 'j_gene': 'j_gene', 'j_allele': 'j_allele',
                 'locus': 'locus', 'tags': 'tags'}
                )
            signal_count = 0
            total_signal_count = 0
            total_rearrangement_count = 0
            for row_number, row in enumerate(old_csv_reader):
                if row['frame_type'] == 'In':
                    l = int(row['cdr3_length'])
                    v_gene = row['v_gene']
                    v_family = row['v_family']
                    j_call = row['j_gene']
                    templates = int(row['templates'])
                    total_rearrangement_count += templates

                    if ((str(v_family) != '') & (str(j_call) != '') & (str(l) != '')):

                        # prepare families
                        if (v_family[5:7].isnumeric()):
                            if (v_family[5:6] == '0'):
                                f = v_family[6:7]
                            else:
                                f = v_family[5:7]

                        else:
                            continue

                        # prepare V genes
                        start = v_gene.find("-") + 1
                        if (start == 0):
                            continue
                        elif (v_gene[start + 1].isnumeric()):
                            if (v_gene[start:start + 1] == '0'):
                                v = v_gene[start + 1:start + 2]
                            else:
                                v = v_gene[start:start + 2]
                        elif (v_gene[start].isnumeric()):
                            v = v_gene[start:start + 1]
                        else:
                            continue
                        # prepare Jgenes
                        if (j_call[9:10].isnumeric()):
                            j = int(j_call[9:10])
                        else:
                            continue

                        key = str(l) + '_' + str(f) + '_' + str(j) + '_' + str(v)
                        if ((key in args.prime_keys) and (ill_flag == True)):
                            tag, rearrangement, amino_acid = implementSignal(args, row['rearrangement'], row['amino_acid'])
                            if tag:
                                signal_count += 1
                                total_signal_count += templates
                            new_csv_reader.writerow(
                                {'rearrangement': rearrangement, 'amino_acid': amino_acid,
                                 'frame_type': row['frame_type'],
                                 'templates': row['templates'], 'cdr3_length': row['cdr3_length'],
                                 'v_family': row['v_family'], 'v_gene': row['v_gene'], 'v_allele': row['v_allele'],
                                 'j_family': row['j_family'], 'j_gene': row['j_gene'], 'j_allele': row['j_allele'],
                                 'locus': row['locus'], 'tags': tag})

                        else:
                            new_csv_reader.writerow(
                                {'rearrangement': row['rearrangement'], 'amino_acid': row['amino_acid'],
                                 'frame_type': row['frame_type'],
                                 'templates': row['templates'], 'cdr3_length': row['cdr3_length'],
                                 'v_family': row['v_family'], 'v_gene': row['v_gene'], 'v_allele': row['v_allele'],
                                 'j_family': row['j_family'], 'j_gene': row['j_gene'], 'j_allele': row['j_allele'],
                                 'locus': row['locus']})

            sample_tags = row['sample_tags']
            if ('Cytomegalovirus +' in sample_tags):
                cmv = 'Cytomegalovirus +'
            elif ('Cytomegalovirus -' in sample_tags):
                cmv = 'Cytomegalovirus -'
            else:
                print(filename + ': unknown tag')

            metadata_info = {'filename': filename, 'subject_id': filename[0:-4], 'ill_status': ill_flag, 'sample_tags': cmv,
                 'rearrangement_count': str(row_number), 'total_rearrangement_count': str(total_rearrangement_count),
                 'signal_count': str(signal_count), 'total_signal_count': str(total_signal_count),
                 'signal_percent': str(round(signal_count / row_number, 6)),
                 'total_signal_percent': str(round(total_signal_count / total_rearrangement_count, 6))}
            #
            # metadata_csv_writer.writerow(
            #     {'filename': filename, 'subject_id': filename[0:-4], 'ill_status': ill_flag, 'sample_tags': cmv,
            #      'rearrangement_count': str(row_number), 'total_rearrangement_count': str(total_rearrangement_count),
            #      'signal_count': str(signal_count), 'total_signal_count': str(total_signal_count),
            #      'signal_percent': str(round(signal_count / row_number, 6)),
            #      'total_signal_percent': str(round(total_signal_count / total_rearrangement_count, 6))})

    print(filename)
    return metadata_info

def simulate(args):
    file_list = []

    for f in os.listdir(args.cmv_path):
        filename = os.fsdecode(f)
        if ((filename.endswith('.tsv')) and (filename not in Known_exclude_list)):
            file_size = int((os.stat(args.cmv_path + filename).st_size - 128) / 8)
            file_list.append((filename, file_size))
    file_list.sort(key=lambda x: x[1])

    metadata_csv_writer = csv.DictWriter(open(args.tsv_path + 'metadata.csv', 'w'), delimiter='\t', fieldnames=['filename','subject_id','ill_status','sample_tags','rearrangement_count','total_rearrangement_count','signal_count','total_signal_count', 'signal_percent', 'total_signal_percent'])
    metadata_csv_writer.writerow({'filename': 'filename', 'subject_id': 'subject_id', 'ill_status': 'ill_status', 'sample_tags': 'sample_tags', 'rearrangement_count': 'rearrangement_count', 'total_rearrangement_count': 'total_rearrangement_count', 'signal_count': 'signal_count','total_signal_count':'total_signal_count', 'signal_percent':'signal_percent', 'total_signal_percent':'total_signal_percent'})

    result = []
    ill_flag = False
    for filename, _ in file_list:
        if ((filename.endswith('.tsv')) and (filename[0:-4] not in Known_exclude_list)):
            result.append(simulate_one_file.remote(args, filename, ill_flag))
            if ill_flag == True:
                ill_flag = False
            else:
                ill_flag = True

    for r in result:
        ray.wait([r])
        metadata_info = ray.get(r)
        metadata_csv_writer.writerow(metadata_info)



# old version before ray
# def simulate(args):
#     file_list = []
#     # fieldnames = ['rearrangement', 'amino_acid', 'frame_type', 'rearrangement_type', 'templates', 'reads', 'frequency', 'productive_frequency', 'cdr3_length', 'v_family', 'v_gene', 'v_allele', 'd_family', 'd_gene', 'd_allele', 'j_family', 'j_gene', 'j_allele', 'v_deletions', 'd5_deletions', 'd3_deletions', 'j_deletions', 'n2_insertions', 'n1_insertions', 'v_index', 'n1_index', 'n2_index', 'd_index', 'j_index', 'v_family_ties', 'v_gene_ties', 'v_allele_ties', 'd_family_ties', 'd_gene_ties', 'd_allele_ties', 'j_family_ties', 'j_gene_ties', 'j_allele_ties', 'sequence_tags', 'v_shm_count', 'v_shm_indexes', 'antibody', 'sample_name', 'species', 'locus', 'product_subtype', 'kit_pool', 'total_templates', 'productive_templates', 'outofframe_templates', 'stop_templates', 'dj_templates', 'total_rearrangements', 'productive_rearrangements', 'outofframe_rearrangements' , 'stop_rearrangements', 'dj_rearrangements', 'total_reads', 'total_productive_reads', 'total_outofframe_reads', 'total_stop_reads', 'total_dj_reads', 'productive_clonality', 'productive_entropy', 'sample_clonality', 'sample_entropy', 'sample_amount_ng', 'sample_cells_mass_estimate', 'fraction_productive_of_cells_mass_estimate', 'sample_cells', 'fraction_productive_of_cells', 'max_productive_frequency', 'max_frequency', 'counting_method', 'primer_set', 'release_date', 'sample_tags', 'fraction_productive', 'order_name', 'kit_id', 'total_t_cells']
#     fieldnames = ['rearrangement', 'amino_acid', 'frame_type', 'templates', 'cdr3_length', 'v_family', 'v_gene','v_allele', 'j_family', 'j_gene', 'j_allele', 'locus', 'tags']
#
#     for f in os.listdir(args.cmv_path):
#         filename = os.fsdecode(f)
#         if ((filename.endswith('.tsv')) and (filename not in Known_exclude_list)):
#             file_size = int((os.stat(args.cmv_path + filename).st_size - 128) / 8)
#             file_class = file_info(filename, file_size)
#             file_list.append(file_class)
#     file_list.sort(key=lambda x: x.file_size)
#
#     metadata_csv_writer = csv.DictWriter(open(args.tsv_path + 'metadata.csv', 'w'), dialect="excel", fieldnames=['filename','subject_id','ill_status','sample_tags','rearrangement_count','total_rearrangement_count','signal_count','total_signal_count', 'signal_percent', 'total_signal_percent'])
#     metadata_csv_writer.writerow({'filename': 'filename', 'subject_id': 'subject_id', 'ill_status': 'ill_status', 'sample_tags': 'sample_tags', 'rearrangement_count': 'rearrangement_count', 'total_rearrangement_count': 'total_rearrangement_count', 'signal_count': 'signal_count','total_signal_count':'total_signal_count', 'signal_percent':'signal_percent', 'total_signal_percent':'total_signal_percent'})
#
#     ill_flag = False
#     for c in file_list:
#         with open(args.cmv_path + c.filename) as old_file:
#             old_csv_reader = csv.DictReader(old_file, dialect="excel-tab")
#             with open(args.tsv_path + c.filename, 'w') as new_file:
#                 new_csv_reader = csv.DictWriter(new_file, dialect="excel-tab", fieldnames=fieldnames, delimiter='\t')
#                 # new_csv_reader.writerow({'rearrangement': 'rearrangement', 'amino_acid': 'amino_acid', 'frame_type': 'frame_type', 'rearrangement_type': 'rearrangement_type', 'templates': 'templates', 'reads': 'reads', 'frequency': 'frequency', 'productive_frequency': 'productive_frequency', 'cdr3_length': 'cdr3_length', 'v_family': 'v_family', 'v_gene': 'v_gene', 'v_allele': 'v_allele', 'd_family': 'd_family', 'd_gene': 'd_gene', 'd_allele': 'd_allele', 'j_family': 'j_family', 'j_gene': 'j_gene', 'j_allele': 'j_allele', 'v_deletions': 'v_deletions', 'd5_deletions': 'd5_deletions', 'd3_deletions': 'd3_deletions', 'j_deletions': 'j_deletions', 'n2_insertions': 'n2_insertions', 'n1_insertions': 'n1_insertions', 'v_index': 'v_index', 'n1_index': 'n1_index', 'n2_index': 'n2_index', 'd_index': 'd_index', 'j_index': 'j_index', 'v_family_ties': 'v_family_ties', 'v_gene_ties': 'v_gene_ties', 'v_allele_ties': 'v_allele_ties', 'd_family_ties': 'd_family_ties', 'd_gene_ties': 'd_gene_ties', 'd_allele_ties': 'd_allele_ties', 'j_family_ties': 'j_family_ties', 'j_gene_ties': 'j_gene_ties', 'j_allele_ties': 'j_allele_ties', 'sequence_tags': 'sequence_tags', 'v_shm_count': 'v_shm_count', 'v_shm_indexes': 'v_shm_indexes', 'antibody': 'antibody', 'sample_name': 'sample_name', 'species': 'species', 'locus': 'locus', 'product_subtype': 'product_subtype', 'kit_pool': 'kit_pool', 'total_templates': 'total_templates', 'productive_templates': 'productive_templates', 'outofframe_templates': 'outofframe_templates', 'stop_templates': 'stop_templates', 'dj_templates': 'dj_templates', 'total_rearrangements': 'total_rearrangements', 'productive_rearrangements': 'productive_rearrangements', 'outofframe_rearrangements': 'outofframe_rearrangements', 'stop_rearrangements': 'stop_rearrangements', 'dj_rearrangements': 'dj_rearrangements', 'total_reads': 'total_reads', 'total_productive_reads': 'total_productive_reads', 'total_outofframe_reads': 'total_outofframe_reads', 'total_stop_reads': 'total_stop_reads', 'total_dj_reads': 'total_dj_reads', 'productive_clonality': 'productive_clonality', 'productive_entropy': 'productive_entropy', 'sample_clonality': 'sample_clonality', 'sample_entropy': 'sample_entropy', 'sample_amount_ng': 'sample_amount_ng', 'sample_cells_mass_estimate': 'sample_cells_mass_estimate', 'fraction_productive_of_cells_mass_estimate': 'fraction_productive_of_cells_mass_estimate', 'sample_cells': 'sample_cells', 'fraction_productive_of_cells': 'fraction_productive_of_cells', 'max_productive_frequency': 'max_productive_frequency', 'max_frequency': 'max_frequency', 'counting_method': 'counting_method', 'primer_set': 'primer_set', 'release_date': 'release_date', 'sample_tags': 'sample_tags', 'fraction_productive': 'fraction_productive', 'order_name': 'order_name', 'kit_id': 'kit_id', 'total_t_cells': 'total_t_cells'}
#                 new_csv_reader.writerow({'rearrangement': 'rearrangement', 'amino_acid': 'amino_acid', 'frame_type': 'frame_type', 'templates': 'templates', 'cdr3_length': 'cdr3_length', 'v_family': 'v_family', 'v_gene': 'v_gene', 'v_allele':'v_allele', 'j_family':'j_family', 'j_gene': 'j_gene', 'j_allele':'j_allele', 'locus':'locus', 'tags': 'tags'}
# )
#                 signal_count = 0
#                 total_signal_count = 0
#                 total_rearrangement_count = 0
#                 for row_number, row in enumerate(old_csv_reader):
#                     if row['frame_type'] == 'In':
#                         l = int(row['cdr3_length'])
#                         v_gene = row['v_gene']
#                         v_family = row['v_family']
#                         j_call = row['j_gene']
#                         templates = int(row['templates'])
#                         total_rearrangement_count += templates
#
#                         if ((str(v_family) != '') & (str(j_call) != '') & (str(l) != '')):
#
#                             # prepare families
#                             if (v_family[5:7].isnumeric()):
#                                 if (v_family[5:6] == '0'):
#                                     f = v_family[6:7]
#                                 else:
#                                     f = v_family[5:7]
#
#                             else:
#                                 continue
#
#                             # prepare V genes
#                             start = v_gene.find("-") + 1
#                             if (start == 0):
#                                 continue
#                             elif (v_gene[start + 1].isnumeric()):
#                                 if (v_gene[start:start + 1] == '0'):
#                                     v = v_gene[start + 1:start + 2]
#                                 else:
#                                     v = v_gene[start:start + 2]
#                             elif (v_gene[start].isnumeric()):
#                                 v = v_gene[start:start + 1]
#                             else:
#                                 continue
#                             # prepare Jgenes
#                             if (j_call[9:10].isnumeric()):
#                                 j = int(j_call[9:10])
#                             else:
#                                 continue
#
#                             key = str(l) + '_' + str(f) + '_' + str(j) + '_' + str(v)
#                             if ((key == args.prime_key) and (ill_flag == True)):
#                                 tag, rearrangement, amino_acid = implementSignal(row['rearrangement'], row['amino_acid'])
#                                 if tag:
#                                     signal_count += 1
#                                     total_signal_count += templates
#                                 new_csv_reader.writerow(
#                                     {'rearrangement': rearrangement, 'amino_acid': amino_acid,
#                                      'frame_type': row['frame_type'],
#                                      'templates': row['templates'], 'cdr3_length': row['cdr3_length'],
#                                      'v_family': row['v_family'], 'v_gene': row['v_gene'], 'v_allele': row['v_allele'],
#                                      'j_family': row['j_family'], 'j_gene': row['j_gene'], 'j_allele': row['j_allele'], 'locus': row['locus'] , 'tags': tag})
#
#                             else:
#                                 new_csv_reader.writerow(
#                                     {'rearrangement': row['rearrangement'], 'amino_acid': row['amino_acid'],
#                                      'frame_type': row['frame_type'],
#                                      'templates': row['templates'], 'cdr3_length': row['cdr3_length'],
#                                      'v_family': row['v_family'], 'v_gene': row['v_gene'], 'v_allele': row['v_allele'],
#                                      'j_family': row['j_family'], 'j_gene': row['j_gene'], 'j_allele': row['j_allele'], 'locus': row['locus']})
#
#                 sample_tags = row['sample_tags']
#                 if ('Cytomegalovirus +' in sample_tags):
#                     cmv = 'Cytomegalovirus +'
#                     c.cmv = True
#                 elif ('Cytomegalovirus -' in sample_tags):
#                     cmv = 'Cytomegalovirus -'
#                     c.cmv = False
#                 else:
#                     print(c.filename + ': unknown tag')
#
#                 metadata_csv_writer.writerow({'filename': c.filename, 'subject_id': c.filename[0:-4], 'ill_status': ill_flag, 'sample_tags': cmv,'rearrangement_count':str(row_number),'total_rearrangement_count':str(total_rearrangement_count), 'signal_count':str(signal_count), 'total_signal_count':str(total_signal_count), 'signal_percent':str(round(signal_count/row_number,6)), 'total_signal_percent':str(round(total_signal_count/total_rearrangement_count,6))})
#
#                 if ill_flag == True:
#                     c.ill_status = True
#                     ill_flag = False
#                 else:
#                     c.ill_status = False
#                     ill_flag = True
#
#                 c.rearrangement_count = row_number
#                 c.total_rearrangement_count = total_rearrangement_count
#                 c.signal_count = signal_count
#                 c.total_signal_count = total_signal_count
#                 print(c.filename)

default_data_path_cmv = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/cmv2/emerson-2017-natgen/'
default_dest_path_cmv = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/simulated_data1/'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--cmv_path', help='cmv_path', type=str, default=default_data_path_cmv)
    parser.add_argument('--tsv_path', help='tsv_path', type=str, default=default_dest_path_cmv)
    # parser.add_argument('--prime_key', help='prime_key', type=str, default='45_6_3_5')
    parser.add_argument('--prime_keys', help='prime_key', type=str, nargs='+', default=['45_6_1_5', '45_5_1_1', '42_6_1_5', '45_6_3_5', '42_6_5_5'])
    parser.add_argument('--signal', help='signal', type=str, default='AAAAAAAAAAAA')
    parser.add_argument('--implementing_rate', help='probability to implant a motive in sequence', type=float, default=0.8)
    parser.add_argument('--HD_probabilities', help='probability to each nucleotide in the motive to be implanted', type=float, default=0.8)
    parser.add_argument('--position_range', help='range probability from which position to start the motive', type=int, default=2)
    parser.add_argument('--threshold', help='minimum appearance for a clone in all the repertories', type=int, default=500)
    args = parser.parse_args()

    Path(args.tsv_path).mkdir(parents=True, exist_ok=True)

    # print('start analyze data')
    # analyze(args)
    print('start analyze_clones_appearance_and_average_sizes')
    analyze_clones_appearance_and_average_sizes(args)
    # print('start simulate data')
    # simulate(args)
    # print('end')
