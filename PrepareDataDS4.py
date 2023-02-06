# source /private/tools/ig-env/bin/activate

import pandas as pd
import os
import pickle
import math
import sklearn.cluster
import numpy as np
import shutil
import changeo.Gene
import xlrd
import random
from sklearn.model_selection import train_test_split

################################################################################
### read all cmv files and create new files to feed the classification model ###
################################################################################

## read all CMV files after genotype(such as: /work/peresay/vdjbase_tcr_projects/DS4/HIP01160/HIP01160_genotyped.tab)
## read all the files from folders that start in H / K and the files are end with: "_genotyped.tab"

path_dataset = '/work/peresay/vdjbase_tcr_projects/DS4'
dest_folder = '/localdata/amit/DNN_classification/DS4'
dest_folder_filter = '/localdata/amit/DNN_classification/DS4_filter'
dest_folder_filter_dsi = '/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/DS4_filter/'

#####

cmv_folders = os.listdir(path_dataset)
cmv_folders = [x for x in cmv_folders if 'HIP' in x or 'Keck' in x]

# metadata = pd.read_excel(r'/localdata/amit/DNN_classification/DS4.xls', sheet_name = 'Subjects')
metadata = pd.read_excel('/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/DS4.xls', sheet_name='Subjects')
metadata_unknown = metadata[metadata["Health Status"].str.contains("Unknown")==True]
metadata_unknown = metadata_unknown['Old name'].to_list() ## find the files that have unknown labels
metadata = metadata[metadata["Health Status"].str.contains("Unknown")==False] # filter repertoires with an unknown label
metadata['cmv_status'] = metadata['Health Status'].replace(['CMV+','CMV-'],['positive','negative'])

for i in range(len(cmv_folders)):
    print (i)
    all_files = os.listdir(path_dataset + '/' + cmv_folders[i])
    genotype_file = [x for x in all_files if '_genotyped.tab' in x]
    shutil.copyfile(path_dataset + '/' + cmv_folders[i] + '/' + genotype_file[0], dest_folder + '/' + genotype_file[0])

## read CMV files, sample from each repertoire and convert each repertoire to a tensor:

cmv_genotype_folders = os.listdir(dest_folder)

for f in range(len(cmv_genotype_folders)):
    data_file = pd.read_csv(dest_folder + '/' + cmv_genotype_folders[f], sep="\t")
    data_file = data_file[data_file.productive == True] # remove unfunctional sequences
    data_file = data_file[["sequence","v_call","d_call","j_call","junction","junction_aa","junction_length","duplicate_count"]]
    data_file['v_gene'] = data_file['v_call'].apply(lambda x: changeo.Gene.getGene(x, action='first'))
    data_file['j_gene'] = data_file['j_call'].apply(lambda x: changeo.Gene.getGene(x, action='first'))
    data_file.to_csv(dest_folder_filter +'/' + cmv_genotype_folders[f], sep="\t")
    print(f)

#######################################################################################################################################################################
### sample 10,000 sequences from each repertoire and arrange them. Then, create tensors of (10,000 , 87 , 4) to  fed the classification model (main_classifier_new) ###
#######################################################################################################################################################################

filter_files = os.listdir(dest_folder_filter_dsi)
matching = [s for s in filter_files if any(xs in s for xs in metadata_unknown)]
filter_files = list(set(filter_files)-set(matching)) ## remove files with unknown labels
filter_files2 = [s[:s.find("_")] for s in filter_files]

N1 = 10000 # number of sequences to sample from each repertoire
N2 = 30000 # number of sequences to sample from each repertoire

def sample_repertoire(num, f, i):
    path = "/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/ordered_data/ordered_data_" + str(num) + "_" + str(f + 1)
    if len(temp) > num:
        temp1 = temp.sample(n=num)  ## sample 10000 sequences from each repertoire
        temp1 = temp1.sort_values(['junction_length', 'v_gene', 'j_gene'], ascending=[True, True, True])  # try to also sort by: v_gene/v_family, j_gene, cdr3_length
        nuc_rearrangement = np.zeros((num, 87, 4), dtype=np.int8)  # define the tensor
        nuc_switcher = {"A": 0, "C": 1, "G": 2, "T": 3}
        for row_number in range(len(temp1)):
            for j, char in enumerate(temp1.sequence.iloc[row_number]):
                if char == 'N':
                    continue
                nuc_rearrangement[row_number, j, nuc_switcher[char]] = 1  # create a one-hot vector of the nucluetides
        sub_metadata = metadata[metadata['Old name'] == filter_files2[i]]
        np.save(path + '/' + sub_metadata['cmv_status'].iloc[0] + '_' + sub_metadata['Old name'].iloc[0] + '.npy',nuc_rearrangement)  ## save the filter datasets

## create 5 different folders for each sampling with N = 10000 / 30000
for f in range(5):
    path1 = "/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/ordered_data/ordered_data_10000_" + str(f+1)
    path2 = "/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/ordered_data/ordered_data_30000_" + str(f+1)
    os.mkdir(path1)
    os.mkdir(path2)

for i in range(len(filter_files)):
# for i in range(396,761):
    print(i)
    temp = pd.read_csv(dest_folder_filter_dsi+filter_files[i], sep="\t")
    temp['duplicate_count'] = temp['duplicate_count'].replace(-1,1) # DeepRC changed the number of sequence counts per repertoire from 􀀀1 to 1 for 3 sequences
    temp = pd.DataFrame(temp.values.repeat(temp.duplicate_count, axis=0), columns=temp.columns) ### replicate the sequences with duplicate_count >1 (by the duplicate_count values)
    for f in range(5):
        sample_repertoire(10000, f, i)
        sample_repertoire(30000, f, i)

