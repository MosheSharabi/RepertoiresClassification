import pandas as pd
import os
import pickle
import math
import sklearn.cluster
import numpy as np

################################################################################
### read all cmv files and create new files to feed the classification model ###
################################################################################

path_dataset = '/home/mnt/dsi_vol1/shared/moshe/datasets/cmv/emerson-2017-natgen/'
path_filter_dataset = '/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/cmv_data/'
# path_rearrange_cdr3 = '/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/ordered_data_10000/'
path_rearrange_cdr3 = '/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/ordered_data_30000/'

cmv_files = os.listdir(path_dataset)

##### remove the files which DeepRC removed:

metadata = pd.read_csv(path_dataset+'metadata.csv', sep="\t")
metadata = metadata.loc[metadata['sample_tags'].str.contains("\+|\-", case=False)] # remove datasets without a label (Cytomegalovirus - / Cytomegalovirus +)
saved_files = metadata['filename'].tolist()

cmv_files = [ele for ele in cmv_files if ele in saved_files] ## cmv saved files - 687 files. DeepRC has 686 repertoires (312 + and 374 - and I have 312 + and 375 -)

###### read the csv files:
for i in range(len(cmv_files)):
    temp = pd.read_csv(path_dataset+cmv_files[i], sep='\t')
    temp = temp[["rearrangement","v_family","v_gene","j_gene","cdr3_length","templates"]]
    temp = temp.dropna()  ## remove rows with nan values
    temp = temp[temp.j_gene != 'unresolved']
    temp = temp[temp.v_gene != 'unresolved']
    temp['v_j_jl'] = temp['v_family'].str[4:]+'_'+temp['j_gene'].str[4:7]+'_'+temp['cdr3_length'].astype(str)
    v_j_jl_freq = temp['v_j_jl'] .value_counts()/len(temp) # calculate v_j_jl usage and save a matrix with the v_j_jl usage of each dataset
    v_j_jl_freq = v_j_jl_freq.to_frame().transpose()
    v_j_jl_freq['filename'] = cmv_files[i]
    if i == 0:
        v_j_jl_freq_df = v_j_jl_freq
    else:
        v_j_jl_freq_df = pd.merge(v_j_jl_freq_df, v_j_jl_freq, how="outer")
    temp.to_csv(path_filter_dataset+cmv_files[i], sep="\t") ## save the filter datasets
    print(i)
#    print(len(temp.index))

##### save data as pickle
pickle.dump(v_j_jl_freq_df,open(path_filter_dataset+"v_j_jl_freq_df.pkl","wb"))

###############################################################################################
### divide the data to K-folds with similar size of repertoire and similar V-J-JL frequency ###
###############################################################################################

##### arrange the repertoires by their size
# metadata.sort_values(by=['rearrangement_count'])

##### load the v_j_jl usage matrix
v_j_jl_freq_df = pickle.load(open(path_filter_dataset+"v_j_jl_freq_df.pkl", "rb"))
v_j_jl_freq_df = v_j_jl_freq_df[v_j_jl_freq_df.columns.drop(list(v_j_jl_freq_df.filter(regex='sol')))]
v_j_jl_freq_df2 = v_j_jl_freq_df.drop(['filename'], axis = 1)
v_j_jl_freq_df2 = v_j_jl_freq_df2.fillna(0)

##### divide the data to math.floor(len(cmv_files)/5) groups by k-means and then choose a representative from each group to divide the data to 5 similar folds
K = 10

kmeans = sklearn.cluster.KMeans(K) #math.floor(len(cmv_files)/5)
kmeans.fit(v_j_jl_freq_df2)
v_j_jl_freq_df['clusters'] = kmeans.labels_

freqs = v_j_jl_freq_df.clusters.value_counts()
v_j_jl_freq_df['folds'] = range(len(cmv_files))

for i in range(K):
    indices = v_j_jl_freq_df.index[v_j_jl_freq_df['clusters'] == i].tolist() # cluster = i+1
    l = np.array_split(indices,5)
    for r in range(5):
        v_j_jl_freq_df.folds[l[r]] = "f" + str(r+1)

v_j_jl_freq_df = v_j_jl_freq_df.merge(metadata[["filename", "ill_status", "total_rearrangement_count"]], how="left")
folds_metadata = v_j_jl_freq_df[['filename','ill_status','folds','total_rearrangement_count']]
folds_metadata['cmv_status'] = folds_metadata['ill_status'].replace([True,False],['positive','negative'])

#######################################################################################################################################################################
### sample 10,000 sequences from each repertoire and arrange them. Then, create tensors of (10,000 , 87 , 4) to  fed the classification model (main_classifier_new) ###
#######################################################################################################################################################################

filter_files = os.listdir(path_filter_dataset)
filter_files = list(filter(lambda f: f.endswith('.tsv'), filter_files)) # filter the tsv files

# filter_files.remove('HIP05763.tsv') # remove 'HIP05763.tsv' which has just 1491 sequences (lower than N=10000)

filter_files.remove('HIP05763.tsv') # remove 'HIP05763.tsv' which has just 1491 sequences (lower than N=30000)
filter_files.remove('HIP13753.tsv') # remove 'HIP05763.tsv' which has just 1491 sequences (lower than N=30000)
filter_files.remove('HIP14110.tsv') # remove 'HIP05763.tsv' which has just 1491 sequences (lower than N=30000)

N = 30000 # number of sequences to sample from each repertoire 10000 / 30000
for i in range(len(filter_files)):
    print(i)
    temp = pd.read_csv(path_filter_dataset+filter_files[i], sep="\t")
    temp = temp[temp.j_gene != 'unresolved']
    temp = temp[temp.v_gene != 'unresolved']
    temp = temp[temp.templates != -1] # remove sequences with templates = -1
    temp = pd.DataFrame(temp.values.repeat(temp.templates, axis=0), columns=temp.columns) ### replicate the sequences with templates >1 (by the templates values)
    temp = temp.sample(n = N)
    # np.lexsort((sample_lengths, sample_Vgenes, sample_families, sample_Jgenes))
    temp = temp.sort_values(['cdr3_length', 'v_gene', 'j_gene'], ascending=[True, True, True]) # try to also sort by: v_gene/v_family, j_gene, cdr3_length
    nuc_rearrangement = np.zeros((N, 87, 4), dtype=np.int8) # define the tensor
    nuc_switcher = {"A": 0, "C": 1, "G": 2, "T": 3}
    for row_number in range(len(temp)):
       for j, char in enumerate(temp.rearrangement.iloc[row_number]):
            if char == 'N':
                continue
            nuc_rearrangement[row_number, j, nuc_switcher[char]] = 1 # create a one-hot vector of the nucluetides
    temp2 = folds_metadata[folds_metadata['filename'] == filter_files[i]]
    np.save(path_rearrange_cdr3+temp2.folds.iloc[0]+'_'+temp2.cmv_status.iloc[0]+'_'+filter_files[i]+'.npy', nuc_rearrangement) ## save the filter datasets

