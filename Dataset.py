import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
import numpy as np
from configs import cfg
from time import sleep
import datetime
from utils import generate_all_cubes
import math
import csv

seed_value = cfg.seed
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)

class GeneratorOnline(tf.keras.utils.Sequence):
    def __init__(self, mode, folds, batch_size=1, shuffle=True):
        self.mode = mode
        self.folds = folds
        self.path =  cfg.data_path #r"/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/ordered_data_30000"  ## ordered_data_10000 / ordered_data_30000
        # self.path = cfg.cleaned_path + r"/cubes/dataINorder"  ### the path!!! : /mnt/dsi_vol1/shared/moshe/datasets/cmv/Cleaned_32000_87_templates_folds/cubes/dataINorder , shape: (32000, 87, 4)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x, self.y = self.get_names_and_labels()
        self.on_epoch_end()
        self.zeros =np.zeros((self.batch_size,cfg.dim,cfg.dim,3,4),dtype=np.int8)
        self.files_names_dic = self.get_files_names_dic()

    def get_files_names_dic(self):
        files_names_dic = {}
        for file in self.x :
            if not (file[file.find("_"):] in files_names_dic):
                temp_list = []
                for file1 in self.x:
                    if (file1[file1.find("_"):] == file[file.find("_"):]):
                        temp_list.append(file1)
                files_names_dic[file[file.find("_"):]] = temp_list
        return files_names_dic

    def get_names_and_labels(self):
        names_list = []
        lable_list = []

        for file_name in os.listdir(self.path):
            if file_name in self.folds: # generator holds file only from the requested folds. self.folds = X_train / X_val / X_test
                names_list.append(file_name)
                if ("positive" in file_name):   ## positive or ill
                    lable_list.append(1)
                else:
                    lable_list.append(0)
        return names_list, np.array(lable_list)

    def on_epoch_end(self):
        # print('Updates indexes after each epoch')
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def files_num(self):
        return math.ceil(len(self.x))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size] ## choose 4 indices (for batch_size=4)
        if(len(indexes) != self.batch_size):
            indexes = self.indexes[-self.batch_size: len(self.indexes)]
        y = np.array(self.y[indexes])
        x = np.array([np.load(self.path + '/' + self.x[i]) for i in indexes])
        return x.astype(np.float32), y  ## return x and y for a specific batch to run on them the sequential model (21_3)

    def getitem_with_fileName(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        if (len(indexes) != self.batch_size):
            # print("batch_size not match")
            indexes = np.random.choice(self.indexes, self.batch_size)
        y = np.array(self.y[indexes])

        x = np.array([np.load(self.path + '/' + self.x[i]) for i in indexes])
        return x, y, [self.x[z] for z in indexes]

    def next_batch_gen(self):
        for i in range(len(self)):
            indexes = self.indexes[i * self.batch_size:(i + 1) * self.batch_size]
            y = np.array(self.y[indexes])
            x = np.array([np.load(self.path + '/' + self.x[i]) for i in indexes])
            yield x, y

    # function for cmv check
    def get_one(self, idx):
        index = self.indexes[idx]
        y = np.array(self.y[index])
        x = np.array([np.load(self.path + '/' + self.x[index])])
        return x, y, self.x[index]
