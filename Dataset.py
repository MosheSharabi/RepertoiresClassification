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

def indexMask3D(arr, mask1, mask2):
    newArr = np.array([np.zeros(len(mask2))], np.int32)  # bug
    newArr = np.delete(newArr, 0, axis=0)  # bug

    a11 = datetime.datetime.now()

    # newArr = np.array([])
    # newArr = np.expand_dims(newArr, axis=0)
    for i in range(len(mask1)):
        [j, k] = mask1[i]
        z = arr[j, k, :][mask2]
        newArr = np.concatenate((newArr, [z]), axis=0)

    b11 = datetime.datetime.now()

    # print(len(mask1))
    return newArr, (b11 - a11).total_seconds()


def fastindexMask3D(arr, mask1, mask2):
    l = list(mask1.transpose())
    newArr = arr[tuple(l)]
    newArr = np.take(newArr, mask2, axis=1)

    # print(len(mask1))
    return newArr


def generate_all_cubes(path):
    all_cubes = {}
    all_labels = {}
    i = 0
    for cube in os.listdir(path):
        if (cube.endswith('.npy')):
            if (cfg.dataset == 'celiac'):
                i = int(cube[cube.find("I") + 1: cube.find("S") - 1])
                all_cubes[i] = np.load(path + '/' + cube)
                if ("ill" in cube):
                    all_labels[i] = 1
                else:
                    all_labels[i] = 0
            elif (cfg.dataset == 'cmv'):
                all_cubes[i] = np.load(path + '/' + cube)
                if ("ill" in cube):
                    all_labels[i] = 1
                else:
                    all_labels[i] = 0
                i += 1
            elif (cfg.dataset == 'biomed'):
                all_cubes[i] = np.load(path + '/' + cube)
                if ("SC" in cube):
                    all_labels[i] = 2
                elif ("HC" in cube):
                    all_labels[i] = 1
                elif ("H" in cube):
                    all_labels[i] = 0
                else:
                    print("EROR in generate_all_cubes in utils")
                i += 1
    return all_cubes, all_labels


class cubeGenerator_online(object):
    def __init__(self, path):
        print("data loaded from:    ", path)
        self.path = path
        self.random_files0 = self.random_files_func()
        self.random_files1 = self.random_files_func()
        self.random_files2 = self.random_files_func()
        self.random_files3 = self.random_files_func()
        self.random_files4 = self.random_files_func()
        self.random_files5 = self.random_files_func()
        self.random_files6 = self.random_files_func()
        self.random_files7 = self.random_files_func()

    def random_files_func(self):
        names_list = []
        for file in os.listdir(self.path):
            if (file.endswith('.npy')):
                names_list.append(file)
        random.shuffle(names_list)
        return names_list

    def get_next_samples(self, gen_name):
        if (gen_name == b'Gen_0'):
            random_files = self.random_files0
        elif (gen_name == b'Gen_1'):
            random_files = self.random_files1
        elif (gen_name == b'Gen_2'):
            random_files = self.random_files2
        elif (gen_name == b'Gen_3'):
            random_files = self.random_files3
        elif (gen_name == b'Gen_4'):
            random_files = self.random_files3
        elif (gen_name == b'Gen_5'):
            random_files = self.random_files3
        elif (gen_name == b'Gen_6'):
            random_files = self.random_files3
        elif (gen_name == b'Gen_7'):
            random_files = self.random_files3

        for file in random_files:
            sleep(0.3)
            if (cfg.dataset == 'cmv'):
                # print(gen_name, "  :   ", file)
                data = np.load(self.path + '/' + file)
                if ("ill" in file):
                    label = 1
                else:
                    label = 0
                yield data, label


class cubeGenerator_online_AE(object):
    def __init__(self, path):
        self.path = path
        self.random_files0 = self.random_files_func()
        self.random_files1 = self.random_files_func()
        self.random_files2 = self.random_files_func()
        self.random_files3 = self.random_files_func()

    def random_files_func(self):
        names_list = []
        for file in os.listdir(self.path):
            if (file.endswith('.npy')):
                names_list.append(file)
        random.shuffle(names_list)
        return names_list

    def get_next_samples(self, gen_name):
        if (gen_name == b'Gen_0'):
            random_files = self.random_files0
        elif (gen_name == b'Gen_1'):
            random_files = self.random_files1
        elif (gen_name == b'Gen_2'):
            random_files = self.random_files2
        elif (gen_name == b'Gen_3'):
            random_files = self.random_files3

        for file in random_files:
            sleep(0.3)
            if (cfg.dataset == 'cmv'):
                # print(gen_name, "  :   ", file)
                data = np.load(self.path + '/' + file)
                if ("ill" in file):
                    label = 1
                else:
                    label = 0
                randnums = np.random.randint(0, len(data), cfg.seq_in_samples)
                yield data[randnums], label


class GeneratorRandomFromUnitedFile(tf.keras.utils.Sequence):
    def __init__(self, path, batch_size=2, steps_per_epoch=20, shuffle=True):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.half_batch = batch_size // 2
        self.path = path
        self.steps_per_epoch = steps_per_epoch
        self.seq_in_samples = cfg.seq_in_samples
        self.dim = cfg.dim
        self.ill_seqs, self.healthy_seqs = self.load_data(path)
        self.ill_range, self.healthy_range = range(len(self.ill_seqs)), range(len(self.healthy_seqs))
        self.y = np.zeros(batch_size, dtype=np.int8)
        self.y[0:self.half_batch] = 1
        self.on_epoch_end()


    # def load_data(self, path):
    #     print("load data")
    #     ill_seqs = np.load(path + "/united_repertoires_ill.npy", mmap_mode='r')
    #     healthy_seqs = np.load(path + "/united_repertoires.npy", mmap_mode='r')
    #     print("data loaded")
    #     return ill_seqs, healthy_seqs
    def load_data(self, path):
        print("load data")
        data = np.load(path + "/data.zip")
        ill_seqs, healthy_seqs = data['united_repertoires_ill.npy'], data['united_repertoires.npy']
        print("data loaded")
        return ill_seqs, healthy_seqs

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes_ill = np.arange(len(self.ill_seqs))
        self.indexes_healthy = np.arange(len(self.healthy_seqs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes_ill)
            np.random.shuffle(self.indexes_healthy)

    # def __getitem__(self, idx):
    #     ill_indices = np.random.choice(self.ill_range, size=(self.half_batch, self.seq_in_samples), replace=False)
    #     healthy_indices = np.random.choice(self.healthy_range, size=(self.half_batch, self.seq_in_samples), replace=False)
    #     ill_samples = self.healthy_seqs[ill_indices].reshape((self.half_batch, self.dim,self.dim,84, 4))
    #     healty_samples = self.healthy_seqs[healthy_indices].reshape((self.half_batch, self.dim,self.dim,84, 4))
    #     batch = np.concatenate((ill_samples, healty_samples))
    #     return batch, self.y

    def __getitem__(self, idx):
        indexes_ill = self.indexes_ill[idx * self.half_batch * self.seq_in_samples:(idx + 1) * self.half_batch * self.seq_in_samples]
        indexes_healthy = self.indexes_healthy[idx * self.half_batch * self.seq_in_samples:(idx + 1) * self.half_batch * self.seq_in_samples]
        if(len(indexes_ill) + len(indexes_healthy) != self.batch_size * self.seq_in_samples):
            # print("batch_size not match")
            indexes_ill = np.random.choice(self.indexes_ill, self.half_batch * self.seq_in_samples)
            indexes_healthy = np.random.choice(self.indexes_healthy, self.half_batch * self.seq_in_samples)
        ill_samples = self.healthy_seqs[indexes_ill].reshape((self.half_batch, self.dim, self.dim, 84, 4))
        healty_samples = self.healthy_seqs[indexes_healthy].reshape((self.half_batch, self.dim, self.dim, 84, 4))
        batch = np.concatenate((ill_samples, healty_samples))
        return batch, self.y

    def __len__(self):
        return math.ceil(min(len(self.indexes_ill), len(self.indexes_healthy)) / (self.half_batch * self.seq_in_samples))

class GeneratorOnline(tf.keras.utils.Sequence):
    def __init__(self, mode, folds, batch_size=1, shuffle=True, noised_sample=cfg.noised_sample, augmentation=False, use_tags=False, add_gender_tag=False,AE=False, extra_sample=cfg.extra_sample):
        self.mode = mode
        self.folds = folds
        self.noised_sample = noised_sample
        self.AE = AE # for autoencoder mode - generate (x=x,y=x)
        self.path = cfg.cleaned_path + r"/cubes/dataINorder"
        if (cfg.dataset == 'cmv'):
            if(mode == "train"):
                    self.cohort_path = cfg.cleaned_path + r"/cohort1.tsv"
            elif(mode == "validation"):
                    self.cohort_path = cfg.cleaned_path + r"/cohort2.tsv"
        self.add_gender_tag = add_gender_tag
        self.extra_sample = extra_sample
        self.use_tags = use_tags
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.shuffle = shuffle
        if (cfg.dataset == 'cmv'):
            if (use_tags):
                self.tags_dic= self.get_tags_dic(self.cohort_path)
        if (cfg.use_embedding):
            self.embed_word_size = 9 # in nucleotide units
            self.embed_stride = 3 # embed_stride <= embed_word_size
            self.embeding_dic = self.create_embeding_dic()
        self.x, self.y = self.get_names_and_labels()
        self.on_epoch_end()
        self.zeros =np.zeros((self.batch_size,cfg.dim,cfg.dim,3,4),dtype=np.int8)
        self.files_names_dic = self.get_files_names_dic()

    def create_embeding_dic(self):
        dic = np.zeros((4 ** self.embed_word_size, 4 * self.embed_word_size), dtype=np.int8)
        for i in range(self.embed_word_size) :
            for j in range(4 ** (i+1)) :
                dic[j * (4**(self.embed_word_size - i - 1)):j * (4**(self.embed_word_size - i - 1)) + 4**(self.embed_word_size - i - 1),4 * i + j % 4] = 1
        dic = {tuple(row): i for i, row in enumerate(dic, 1)}
        return dic

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

    def get_tags_dic(self,path):
        tags_dic = {}

        SampleOverview = csv.DictReader(open(path), dialect="excel-tab")

        for r in SampleOverview:
            flag = 1
            virus_status = -1
            gender = -1
            sample_name = r["sample_name"]
            sample_tags = r["sample_tags"]

            if ("Cytomegalovirus +" in sample_tags):
                virus_status = 1
            elif ("Cytomegalovirus -" in sample_tags):
                virus_status = 0
            else:
                flag=0

            if ("Female" in sample_tags):
                gender = 0
            elif ("Male" in sample_tags):
                gender = 1
            else:
                flag = 0

            if(flag):
                tags_dic[sample_name] = (virus_status,gender)

            # ill_male = 0
            # ill_female = 0
            # healthy_male = 0
            # healthy_female = 0
            # for n in tags_dic:
            #     if(tags_dic[n][0] == 1):
            #         if(tags_dic[n][1] == 1):
            #             ill_male += 1
            #         else:
            #             ill_female += 1
            #     else:
            #         if (tags_dic[n][1] == 1):
            #             healthy_male += 1
            #         else:
            #             healthy_female += 1

        return tags_dic




    def get_names_and_labels(self):
        names_list = []
        lable_list = []
        for file_name in os.listdir(self.path):
            start = file_name.find('_') + 1
            if not (file_name[start: start + 2] in self.folds): # generator holds file only from the requested folds
                continue
            if (file_name.endswith('.npy')):
                if ((cfg.dataset == 'celiac') | (cfg.dataset == 'cmv')):
                    names_list.append(file_name)
                    if ("ill" in file_name):
                        lable_list.append(1)
                    else:
                        lable_list.append(0)
                elif (cfg.dataset == 'biomed'):
                    names_list.append(file_name)
                    if ("SC" in file_name):
                        lable_list.append(2)
                    elif ("HC" in file_name):
                        lable_list.append(1)
                    elif ("H" in file_name):
                        lable_list.append(0)
                    else:
                        print("EROR in get_names_and_labels")

        if(self.use_tags):   # cmv only
            gender_list = []
            for name in names_list:
                gender_list.append(self.tags_dic[name[name.find("_") + 1:name.find("_") + 9]][1])
            gender_list = np.array(gender_list)
            lable_list = np.array(lable_list)
            y = np.stack((lable_list, gender_list), axis=1)
            return names_list, y


        if ((cfg.dataset == 'celiac') | (cfg.dataset == 'cmv')):
            return names_list, np.array(lable_list)
        elif (cfg.dataset == 'biomed'):
            return names_list, tf.keras.utils.to_categorical( np.array(lable_list), num_classes=3, dtype='int8')

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def shaffel_augmentation(self,x):
        permute = np.unravel_index(np.random.permutation(cfg.seq_in_samples), (cfg.dim, cfg.dim))
        return x[permute].reshape((cfg.dim, cfg.dim,84,4))

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    def files_num(self):
        return math.ceil(len(self.x))

    def __getitem__(self, idx):
        # return np.zeros((2,32000,87,4),dtype=np.float32), np.zeros((2),dtype=np.int8) # demo sample for tests
        # return [np.zeros((2,32000,87,4),dtype=np.float32), np.ones((2,32000//16,1),dtype=np.float32)], np.zeros((2),dtype=np.int8) # demo sample for tests


        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        if(len(indexes) != self.batch_size):
            # print("batch_size not match")
            # indexes = np.random.choice(self.indexes, self.batch_size)
            indexes = self.indexes[-self.batch_size: len(self.indexes)]
        y = np.array(self.y[indexes])
        x = np.array([np.load(self.path + '/' + self.x[i]) for i in indexes])

        if(self.extra_sample > 1):
            for z in range(self.extra_sample -1):
                twin_samples = []
                for j, i in enumerate(indexes):
                    n = self.x[i]
                    name = self.x[i][self.x[i].find("_"):]
                    twin_name = random.choice(self.files_names_dic[name])
                    twin_samples.append(twin_name)

                x1 = np.array([np.load(self.path + '/' + i) for i in twin_samples])
                x = np.concatenate((x, x1), axis=1)

        if(self.add_gender_tag):
            x = np.concatenate((x, self.zeros), axis=-2)
            for j, i in enumerate(indexes) :
                if(self.tags_dic[self.x[i][self.x[i].find("_") + 1:self.x[i].find("_") + 9]][1]):
                    # x[i1] = np.concatenate(x[i1] , self.ones)
                    x[j, :, :,84:88, 0] = 1

        if(self.augmentation):
            for i in range(self.batch_size):
                z = random.random()
                if(z < 0.1):
                    x[i] = self.shaffel_augmentation(x[i])

        if (self.noised_sample):
            noise = np.random.normal(loc=0, scale=0.01, size=x.shape)
            x = x + noise

        # x = x[:,1000:-8000,:,:]
        if (self.AE):
            x = x[:,np.random.choice(x.shape[1], 8000, replace=False)]
            return x, x

        if (cfg.use_embedding):
            # print(x.shape)
            x = x.reshape(x.shape[0],x.shape[1],-1)
            # print(x.shape)
            tokenized_x = [[[self.embeding_dic[tuple(seq[4*self.embed_stride * i:4*self.embed_stride * i + 4*self.embed_word_size])] for i in range((87 -self.embed_word_size)// self.embed_stride + 1)] for seq in bach] for bach in x]
            tokenized_x = np.array(tokenized_x,dtype=np.int32)
            # print(tokenized_x.shape)
            return tokenized_x, y

        return x.astype(np.float32), y
        # return [x.astype(np.float32), np.ones((self.batch_size,cfg.seq_in_samples//16,1),dtype=np.float32)], y

    def getitem_with_fileName(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        if (len(indexes) != self.batch_size):
            # print("batch_size not match")
            indexes = np.random.choice(self.indexes, self.batch_size)
        y = np.array(self.y[indexes])

        x = np.array([np.load(self.path + '/' + self.x[i]) for i in indexes])

        if(self.extra_sample > 1):
            for z in range(self.extra_sample -1):
                twin_samples = []
                for j, i in enumerate(indexes):
                    n = self.x[i]
                    name = self.x[i][self.x[i].find("_"):]
                    twin_name = random.choice(self.files_names_dic[name])
                    twin_samples.append(twin_name)

                x1 = np.array([np.load(self.path + '/' + i) for i in twin_samples])
                x = np.concatenate((x, x1), axis=1)

        if (self.add_gender_tag):
            x = np.concatenate((x, self.zeros), axis=-2)
            for j, i in enumerate(indexes):
                if (self.tags_dic[self.x[i][self.x[i].find("_") + 1:self.x[i].find("_") + 9]][1]):
                    # x[i1] = np.concatenate(x[i1] , self.ones)
                    x[j, :, :, 84:88, 0] = 1

        if (self.augmentation):
            for i in range(self.batch_size):
                z = random.random()
                if (z < 0.1):
                    x[i] = self.shaffel_augmentation(x[i])
        # x = np.zeros((self.batch_size,32000,87,4),dtype=np.float32)   # demo sample for tests
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

# like the generator above, with one folder and one path with all folds - to erase
#
# class GeneratorOnline(tf.keras.utils.Sequence):
#     def __init__(self, mode, batch_size=1, shuffle=True, noised_sample=cfg.noised_sample, augmentation=False, use_tags=False, add_gender_tag=False, extra_sample=cfg.extra_sample):
#         self.mode = mode
#         self.noised_sample = noised_sample
#         if(mode == "train"):
#             self.path = cfg.cleaned_path + r"/cubes/train/dataINorder"
#             # self.path = cfg.cleaned_path + r"/cubes/train/dataINorder/temp"
#             if (cfg.dataset == 'cmv'):
#                 self.cohort_path = cfg.cleaned_path + r"/cohort1.tsv"
#         elif(mode == "validation"):
#             self.path = cfg.cleaned_path + r"/cubes/validation/dataINorder"
#             # self.path = cfg.cleaned_path + r"/cubes/validation/dataINorder/temp"
#             if (cfg.dataset == 'cmv'):
#                 self.cohort_path = cfg.cleaned_path + r"/cohort2.tsv"
#         self.add_gender_tag = add_gender_tag
#         self.extra_sample = extra_sample
#         self.use_tags = use_tags
#         self.augmentation = augmentation
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         if (cfg.dataset == 'cmv'):
#             if (use_tags):
#                 self.tags_dic= self.get_tags_dic(self.cohort_path)
#         self.x, self.y = self.get_names_and_labels()
#         self.on_epoch_end()
#         self.zeros =np.zeros((self.batch_size,cfg.dim,cfg.dim,3,4),dtype=np.int8)
#         self.files_names_dic = self.get_files_names_dic()
#
#     def get_files_names_dic(self):
#         files_names_dic = {}
#         for file in self.x :
#             if not (file[file.find("_"):] in files_names_dic):
#                 temp_list = []
#                 for file1 in self.x:
#                     if (file1[file1.find("_"):] == file[file.find("_"):]):
#                         temp_list.append(file1)
#                 files_names_dic[file[file.find("_"):]] = temp_list
#         return files_names_dic
#
#     def get_tags_dic(self,path):
#         tags_dic = {}
#
#         SampleOverview = csv.DictReader(open(path), dialect="excel-tab")
#
#         for r in SampleOverview:
#             flag = 1
#             virus_status = -1
#             gender = -1
#             sample_name = r["sample_name"]
#             sample_tags = r["sample_tags"]
#
#             if ("Cytomegalovirus +" in sample_tags):
#                 virus_status = 1
#             elif ("Cytomegalovirus -" in sample_tags):
#                 virus_status = 0
#             else:
#                 flag=0
#
#             if ("Female" in sample_tags):
#                 gender = 0
#             elif ("Male" in sample_tags):
#                 gender = 1
#             else:
#                 flag = 0
#
#             if(flag):
#                 tags_dic[sample_name] = (virus_status,gender)
#
#             # ill_male = 0
#             # ill_female = 0
#             # healthy_male = 0
#             # healthy_female = 0
#             # for n in tags_dic:
#             #     if(tags_dic[n][0] == 1):
#             #         if(tags_dic[n][1] == 1):
#             #             ill_male += 1
#             #         else:
#             #             ill_female += 1
#             #     else:
#             #         if (tags_dic[n][1] == 1):
#             #             healthy_male += 1
#             #         else:
#             #             healthy_female += 1
#
#         return tags_dic
#
#
#
#
#     def get_names_and_labels(self):
#         names_list = []
#         lable_list = []
#         for file_name in os.listdir(self.path):
#             if (file_name.endswith('.npy')):
#                 if ((cfg.dataset == 'celiac') | (cfg.dataset == 'cmv')):
#                     names_list.append(file_name)
#                     if ("ill" in file_name):
#                         lable_list.append(1)
#                     else:
#                         lable_list.append(0)
#                 elif (cfg.dataset == 'biomed'):
#                     names_list.append(file_name)
#                     if ("SC" in file_name):
#                         lable_list.append(2)
#                     elif ("HC" in file_name):
#                         lable_list.append(1)
#                     elif ("H" in file_name):
#                         lable_list.append(0)
#                     else:
#                         print("EROR in get_names_and_labels")
#
#         if(self.use_tags):   # cmv only
#             gender_list = []
#             for name in names_list:
#                 gender_list.append(self.tags_dic[name[name.find("_") + 1:name.find("_") + 9]][1])
#             gender_list = np.array(gender_list)
#             lable_list = np.array(lable_list)
#             y = np.stack((lable_list, gender_list), axis=1)
#             return names_list, y
#
#
#         if ((cfg.dataset == 'celiac') | (cfg.dataset == 'cmv')):
#             return names_list, np.array(lable_list)
#         elif (cfg.dataset == 'biomed'):
#             return names_list, tf.keras.utils.to_categorical( np.array(lable_list), num_classes=3, dtype='int8')
#
#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.x))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#     def shaffel_augmentation(self,x):
#         permute = np.unravel_index(np.random.permutation(cfg.seq_in_samples), (cfg.dim, cfg.dim))
#         return x[permute].reshape((cfg.dim, cfg.dim,84,4))
#
#     def __len__(self):
#         return math.ceil(len(self.x) / self.batch_size)
#     def files_num(self):
#         return math.ceil(len(self.x))
#
#     def __getitem__(self, idx):
#         indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
#         if(len(indexes) != self.batch_size):
#             # print("batch_size not match")
#             # indexes = np.random.choice(self.indexes, self.batch_size)
#             indexes = self.indexes[-self.batch_size: len(self.indexes)]
#         y = np.array(self.y[indexes])
#
#         x = np.array([np.load(self.path + '/' + self.x[i]) for i in indexes])
#
#         if(self.extra_sample > 1):
#             for z in range(self.extra_sample -1):
#                 twin_samples = []
#                 for j, i in enumerate(indexes):
#                     n = self.x[i]
#                     name = self.x[i][self.x[i].find("_"):]
#                     twin_name = random.choice(self.files_names_dic[name])
#                     twin_samples.append(twin_name)
#
#                 x1 = np.array([np.load(self.path + '/' + i) for i in twin_samples])
#                 x = np.concatenate((x, x1), axis=1)
#
#         if(self.add_gender_tag):
#             x = np.concatenate((x, self.zeros), axis=-2)
#             for j, i in enumerate(indexes) :
#                 if(self.tags_dic[self.x[i][self.x[i].find("_") + 1:self.x[i].find("_") + 9]][1]):
#                     # x[i1] = np.concatenate(x[i1] , self.ones)
#                     x[j, :, :,84:88, 0] = 1
#
#         if(self.augmentation):
#             for i in range(self.batch_size):
#                 z = random.random()
#                 if(z < 0.1):
#                     x[i] = self.shaffel_augmentation(x[i])
#
#         if (self.noised_sample):
#             noise = np.random.normal(loc=0, scale=0.01, size=x.shape)
#             x = x + noise
#
#         # x = x[:,1000:-8000,:,:]
#         return x, y
#
#     def getitem_with_fileName(self, idx):
#         indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
#         if (len(indexes) != self.batch_size):
#             # print("batch_size not match")
#             indexes = np.random.choice(self.indexes, self.batch_size)
#         y = np.array(self.y[indexes])
#
#         x = np.array([np.load(self.path + '/' + self.x[i]) for i in indexes])
#
#         if(self.extra_sample > 1):
#             for z in range(self.extra_sample -1):
#                 twin_samples = []
#                 for j, i in enumerate(indexes):
#                     n = self.x[i]
#                     name = self.x[i][self.x[i].find("_"):]
#                     twin_name = random.choice(self.files_names_dic[name])
#                     twin_samples.append(twin_name)
#
#                 x1 = np.array([np.load(self.path + '/' + i) for i in twin_samples])
#                 x = np.concatenate((x, x1), axis=1)
#
#         if (self.add_gender_tag):
#             x = np.concatenate((x, self.zeros), axis=-2)
#             for j, i in enumerate(indexes):
#                 if (self.tags_dic[self.x[i][self.x[i].find("_") + 1:self.x[i].find("_") + 9]][1]):
#                     # x[i1] = np.concatenate(x[i1] , self.ones)
#                     x[j, :, :, 84:88, 0] = 1
#
#         if (self.augmentation):
#             for i in range(self.batch_size):
#                 z = random.random()
#                 if (z < 0.1):
#                     x[i] = self.shaffel_augmentation(x[i])
#         return x, y, [self.x[z] for z in indexes]
#
#     def next_batch_gen(self):
#         for i in range(len(self)):
#             indexes = self.indexes[i * self.batch_size:(i + 1) * self.batch_size]
#             y = np.array(self.y[indexes])
#             x = np.array([np.load(self.path + '/' + self.x[i]) for i in indexes])
#             yield x, y
#     # function for cmv check
#     def get_one(self, idx):
#         index = self.indexes[idx]
#         y = np.array(self.y[index])
#         x = np.array([np.load(self.path + '/' + self.x[index])])
#         return x, y, self.x[index]


class cubeGenerator(object):
    def __init__(self, path):
        self.path = path
        self.all_cubes, self.all_labels = generate_all_cubes(path)

    def get_next_samples(self):
        for i in range(len(self.all_labels)):
            cubeIdx = random.choice(list(self.all_cubes))
            randnums = np.random.randint(0, len(self.all_cubes[cubeIdx]), cfg.seq_in_samples)
            samples = self.all_cubes[cubeIdx][randnums]
            yield samples, self.all_labels[cubeIdx]


class cubeGenerator_update_samples(object):
    def __init__(self, path):
        self.path = path
        self.all_cubes, self.all_labels = self.generate_all_cubes(self.path)
        self.numOfSamples = 1000
        self.counter = 0

    def generate_all_cubes(self, path):
        all_cubes = {}
        all_labels = {}
        for i, cube in enumerate(os.listdir(self.path)):
            all_cubes[i] = np.load(path + '/' + cube)
            if ("ill" in cube):
                all_labels[i] = 1
            else:
                all_labels[i] = 0

        print(len(all_cubes), ' reps')
        return all_cubes, all_labels

    def get_next_samples(self):
        while True:
            self.counter += 1
            if (self.counter == 128):
                self.counter = 0
                self.all_cubes, self.all_labels = self.generate_all_cubes(self.path)

            cubeIdx = random.randrange(len(self.all_cubes))
            samples = self.all_cubes[cubeIdx]
            yield samples, self.all_labels[cubeIdx]


class cubeGenerator_reorder(object):
    def __init__(self, path, model):
        self.path = path
        self.all_cubes, self.all_labels = self.generate_all_cubes(path)
        self.numOfSamples = 1000
        self.model = model

    def reorder(self, samples):
        # print("samples:    ", np.shape(samples))
        ep1 = np.expand_dims(samples, axis=0)
        ep2 = np.expand_dims(ep1, axis=0)
        x = self.model(ep2, isFullWay=False)
        # print(np.shape(x))
        # print(type(x))
        x = x.numpy()
        sq = np.squeeze(x, axis=0)
        # order = self.findOrderNP(sq)
        # order = self.fastfindOrderNP(sq)
        # order = self.fasterfindOrderNP(sq)
        order = self.fastfindOrderNP_Nmin(sq)
        # print(np.shape(order))
        reorder_samples = samples[order]
        # print(np.shape(reorder_samples))
        return reorder_samples

    def findOrderNP(self, x):
        delta11 = 0.0
        delta12 = 0.0
        delta2 = 0.0
        delta3 = 0.0
        # a = datetime.datetime.now()
        DIM = int(np.sqrt(cfg.seq_in_samples))
        # A: contains the new arrangement, Initialized -1 to indicate square is empty
        A = np.zeros(shape=(DIM, DIM), dtype=int) - 1
        # B: contains the distance vectors, Initialized 100 to be far from chosen as minimum
        B = np.zeros(shape=(DIM, DIM, cfg.seq_in_samples), dtype=float) + 100
        # C: Contains the number neighbors, Initialized 0 as there is no neighbors in the beginning
        C = np.zeros(shape=(DIM, DIM), dtype=int)

        # start with random vector
        # tempIndex = np.random.choice(vectorsIn)
        tempIndex = np.random.randint(0, len(x[0]))
        A[DIM // 2, DIM // 2] = tempIndex
        tempVec = x[0, tempIndex]
        disVec = np.mean(np.square(np.subtract(x[0], tempVec)), axis=1)  # DistanceFromAllVectors
        disVec[tempIndex] = 100  # no need to look at this vector anywhere again
        for j in range(DIM // 2 - 1, DIM // 2 + 2):
            for k in range(DIM // 2 - 1, DIM // 2 + 2):
                if (0 <= j < DIM) & (0 <= k < DIM):
                    if (A[j, k] == -1):
                        B[j, k] = disVec
                        C[j, k] += 1

        B[DIM // 2, DIM // 2] = 100  # no need to look at this square again
        B[:, :, tempIndex] = 100  # no need to look at this vector anywhere again

        b = datetime.datetime.now()
        # print(b-a)
        # print((b-a).total_seconds())

        # go for all the rest vectors
        for i in range(len(x[0]) - 1):
            a11 = datetime.datetime.now()

            index = np.unravel_index(np.argmin(B, axis=None), B.shape)  # square and vector with the minimum distance

            b11 = datetime.datetime.now()
            delta11 += (b11 - a11).total_seconds()

            a12 = datetime.datetime.now()

            A[index[0], index[1]] = index[2]
            tempVec = x[0, index[2]]
            disVec = np.mean(np.square(np.subtract(x[0], tempVec)), axis=1)  # DistanceFromAllVectors

            b12 = datetime.datetime.now()
            delta12 += (b12 - a12).total_seconds()

            a2 = datetime.datetime.now()

            # update surrounding squares in B
            for j in range(index[0] - 1, index[0] + 2):
                for k in range(index[1] - 1, index[1] + 2):
                    # check if: (j , k) in matrix boundaries. (j , k) is not corrent vec / (j , k) is empty square in A
                    if (0 <= j < DIM) & (0 <= k < DIM):
                        if (A[j, k] == -1):
                            B[j, k] = (C[j, k] * B[j, k] + disVec) / (C[j, k] + 1)  # update B with relative weight
                            C[j, k] += 1

            b2 = datetime.datetime.now()
            delta2 += (b2 - a2).total_seconds()

            a3 = datetime.datetime.now()

            B[index[0], index[1]] = 100  # no need to look at this square again
            B[:, :, index[2]] = 100  # no need to look at this vector anywhere again

            b3 = datetime.datetime.now()
            delta3 += (b3 - a3).total_seconds()

        print("delta11:    ", delta11)
        print("delta12:    ", delta12)
        print("delta2:    ", delta2)
        print("delta3:    ", delta3)
        print("total:    ", delta11 + delta12 + delta2 + delta3)

        return A

    def fastfindOrderNP(self, x):
        deltaindexMask3D = 0.0
        delta11 = 0.0
        delta12 = 0.0
        delta2 = 0.0
        delta3 = 0.0
        a = datetime.datetime.now()
        DIM = int(np.sqrt(cfg.seq_in_samples))
        # A: contains the new arrangement, Initialized -1 to indicate square is empty
        A = np.zeros(shape=(DIM, DIM), dtype=int) - 1
        # B: contains the distance vectors, Initialized 100 to be far from chosen as minimum
        B = np.zeros(shape=(DIM, DIM, cfg.seq_in_samples), dtype=float) + 100
        # C: Contains the number neighbors, Initialized 0 as there is no neighbors in the beginning
        C = np.zeros(shape=(DIM, DIM), dtype=int)

        # mask1 = np.arange(0)   # holds vectors indices we should search in an argmin in B
        mask1 = np.array([np.zeros(2)], np.int32)  # bug
        mask1 = np.delete(mask1, 0, axis=0)  # bug
        mask2 = np.arange(len(x[0]))  # holds vectors indices we didn't yet position in A

        # start with random vector
        # tempIndex = np.random.choice(vectorsIn)
        tempIndex = np.random.randint(0, len(x[0]))
        mask2 = mask2[mask2 != tempIndex]
        A[DIM // 2, DIM // 2] = tempIndex
        tempVec = x[0, tempIndex]
        disVec = np.mean(np.square(np.subtract(x[0][mask2], tempVec)), axis=1)  # DistanceFromAllVectors (in mask2)
        # disVec[tempIndex] = 100  # no need to look at this vector anywhere again
        for j in range(DIM // 2 - 1, DIM // 2 + 2):
            for k in range(DIM // 2 - 1, DIM // 2 + 2):
                if (0 <= j < DIM) & (0 <= k < DIM):
                    if (A[j, k] == -1):
                        if (C[j, k] == 0):
                            mask1 = np.concatenate((mask1, [[j, k]]), axis=0)
                        C[j, k] += 1
                        B[j, k][mask2] = disVec

        # B[DIM // 2, DIM // 2] = 100  # no need to look at this square again
        # B[:, :, tempIndex] = 100  # no need to look at this vector anywhere again

        b = datetime.datetime.now()
        print("first part:  ", (b - a).total_seconds())

        # go for all the rest vectors
        for i in range(len(x[0]) - 1):
            a11 = datetime.datetime.now()
            smallB = fastindexMask3D(B, mask1, mask2)
            # smallB, delta = indexMask3D(B, mask1, mask2)
            mask_idx = np.unravel_index(np.argmin(smallB, axis=None),
                                        smallB.shape)  # square and vector with the minimum distance
            # index = np.unravel_index(np.argmin(B, axis=None), B.shape)  # square and vector with the minimum distance
            [idx_0, idx_1] = mask1[mask_idx[0]]  # convert mask_idx to real indeces in A/B/C (mask1)
            idx_2 = mask2[mask_idx[1]]  # convert mask_idx to real indeces in x (mask2)

            b11 = datetime.datetime.now()
            delta11 += (b11 - a11).total_seconds()
            a12 = datetime.datetime.now()

            mask1 = np.delete(mask1, mask_idx[0], axis=0)  # remove chozen vec from mask1
            mask2 = np.delete(mask2, mask_idx[1], axis=0)  # remove chozen vec from mask2
            A[idx_0, idx_1] = idx_2
            tempVec = x[0, idx_2]
            disVec = np.mean(np.square(np.subtract(x[0][mask2], tempVec)), axis=1)  # DistanceFromAllVectors (in mask2)
            # disVec = np.mean(np.square(np.subtract(x[0], tempVec)), axis=1)  # DistanceFromAllVectors

            b12 = datetime.datetime.now()
            delta2 += (b12 - a12).total_seconds()

            a2 = datetime.datetime.now()

            # update surrounding squares in B
            for j in range(idx_0 - 1, idx_0 + 2):
                for k in range(idx_1 - 1, idx_1 + 2):
                    # check if: (j , k) in matrix boundaries. (j , k) is not corrent vec / (j , k) is empty square in A
                    if (0 <= j < DIM) & (0 <= k < DIM):
                        if (A[j, k] == -1):
                            if (C[j, k] == 0):
                                mask1 = np.concatenate((mask1, [[j, k]]), axis=0)
                            B[j, k][mask2] = (C[j, k] * B[j, k][mask2] + disVec) / (
                                        C[j, k] + 1)  # update B with relative weight
                            C[j, k] += 1

            b2 = datetime.datetime.now()
            delta2 += (b2 - a2).total_seconds()

            a3 = datetime.datetime.now()

            # B[index[0], index[1]] = 100  # no need to look at this square again
            # B[:, :, index[2]] = 100  # no need to look at this vector anywhere again

            b3 = datetime.datetime.now()
            delta3 += (b3 - a3).total_seconds()

        print("delta11:    ", delta11)
        print("delta12:    ", delta12)
        print("delta2:    ", delta2)
        print("delta3:    ", delta3)
        print("total:    ", delta11 + delta12 + delta2 + delta3)

        return A

    def fastfindOrderNP_Nmin(self, x):
        delta1 = 0.0
        delta2 = 0.0
        delta3 = 0.0
        num_of_seq = cfg.seq_in_samples
        a = datetime.datetime.now()
        DIM = int(np.sqrt(cfg.seq_in_samples))
        # A: contains the new arrangement, Initialized -1 to indicate square is empty
        A = np.zeros(shape=(DIM, DIM), dtype=int) - 1
        # B: contains the distance vectors, Initialized 100 to be far from chosen as minimum
        B = np.zeros(shape=(DIM, DIM, cfg.seq_in_samples), dtype=float) + 100
        # C: Contains the number neighbors, Initialized 0 as there is no neighbors in the beginning
        C = np.zeros(shape=(DIM, DIM), dtype=int)

        # mask1 = np.arange(0)   # holds vectors indices we should search in an argmin in B
        mask1 = np.array([np.zeros(2)], np.int32)  # bug
        mask1 = np.delete(mask1, 0, axis=0)  # bug
        mask2 = np.arange(cfg.seq_in_samples)  # holds vectors indices we didn't yet position in A

        # start with random vector
        # tempIndex = np.random.choice(vectorsIn)
        tempIndex = np.random.randint(0, cfg.seq_in_samples)
        mask2 = mask2[mask2 != tempIndex]
        A[DIM // 2, DIM // 2] = tempIndex
        num_of_seq -= 1
        tempVec = x[0, tempIndex]
        disVec = np.mean(np.square(np.subtract(x[0][mask2], tempVec)), axis=1)  # DistanceFromAllVectors (in mask2)
        # disVec[tempIndex] = 100  # no need to look at this vector anywhere again
        for j in range(DIM // 2 - 1, DIM // 2 + 2):
            for k in range(DIM // 2 - 1, DIM // 2 + 2):
                if (0 <= j < DIM) & (0 <= k < DIM):
                    if (A[j, k] == -1):
                        if (C[j, k] == 0):
                            mask1 = np.concatenate((mask1, [[j, k]]), axis=0)
                        C[j, k] += 1
                        B[j, k][mask2] = disVec

        # B[DIM // 2, DIM // 2] = 100  # no need to look at this square again
        # B[:, :, tempIndex] = 100  # no need to look at this vector anywhere again

        # go for all the rest vectors
        K = 1
        isend = 0
        # for i in range((num_of_seq - 1)//K):
        while (isend == 0):
            column = 0
            mask1_placed_list = []
            mask2_placed_list = []
            # a1 = datetime.datetime.now()
            smallB = fastindexMask3D(B, mask1, mask2)
            # b1 = datetime.datetime.now()
            # delta1 += (b1 - a1).total_seconds()

            # a2 = datetime.datetime.now()
            # smallB, delta = indexMask3D(B, mask1, mask2)
            # num_vec_to_place = max(smallB.shape[0]//2,1)
            # num_vec_to_place = smallB.shape[0]//2
            # print(smallB.shape[0], smallB.shape[0]//2,K)
            K = max(smallB.shape[0] // 4, K)
            if (smallB.shape[0] <= K):
                K = smallB.shape[0]
                if (K == num_of_seq):
                    isend = 1
            num_vec_to_place = K
            # print(K)
            # print(num_vec_to_place,smallB.shape[0], num_of_seq)

            all_K_min_in_rows = np.argpartition(smallB, num_vec_to_place - isend, axis=1)[:, :num_vec_to_place]
            all_K_min_in_value = smallB[np.arange(smallB.shape[0])[:, None], all_K_min_in_rows]  # value = index_2

            # b2 = datetime.datetime.now()
            # delta2 += (b2 - a2).total_seconds()

            # a3 = datetime.datetime.now()
            # num_vec_to_place = min(all_K_min_in_rows.shape[0],K)
            while (num_vec_to_place):
                sorted_rows = np.argsort(all_K_min_in_value[:, column])
                # print(column ,sorted_rows)
                # sorted_rows = sorted_rows[save_for_later_list]
                column += 1
                save_for_later_list = []
                for i_mask_idx_0, mask_idx_0 in enumerate(sorted_rows):
                    mask_idx_1 = all_K_min_in_rows[mask_idx_0][column - 1]
                    if ((mask_idx_0 in mask1_placed_list) | (mask_idx_1 in mask2_placed_list)):
                        save_for_later_list.append(mask_idx_0)
                        # print(mask_idx_0, mask_idx_1)
                        continue
                        # TODO check how much time it happened
                    else:
                        # save_for_later_list = np.delete(save_for_later_list, i_mask_idx_0, axis=0)  # update mask2
                        mask1_placed_list.append(mask_idx_0)
                        mask2_placed_list.append(mask_idx_1)
                        num_vec_to_place -= 1
                        num_of_seq -= 1
                        # print(mask1_placed_list)

                        [idx_0, idx_1] = mask1[mask_idx_0]  # convert mask_idx to real indeces in A/B/C (mask1)
                        idx_2 = mask2[mask_idx_1]  # convert mask_idx to real indeces in x (mask2)

                        A[idx_0, idx_1] = idx_2

                        tempVec = x[0, idx_2]
                        disVec = np.mean(np.square(np.subtract(x[0][mask2], tempVec)),
                                         axis=1)  # DistanceFromAllVectors (in mask2)
                        # disVec = np.mean(np.square(np.subtract(x[0], tempVec)), axis=1)  # DistanceFromAllVectors

                        # update surrounding squares in B
                        for j in range(idx_0 - 1, idx_0 + 2):
                            for k in range(idx_1 - 1, idx_1 + 2):
                                # check if: (j , k) in boundaries. (j , k) is not corrent vec / (j , k) is empty square in A
                                if (0 <= j < DIM) & (0 <= k < DIM):
                                    if (A[j, k] == -1):
                                        if (C[j, k] == 0):
                                            mask1 = np.concatenate((mask1, [[j, k]]), axis=0)
                                        B[j, k][mask2] = (C[j, k] * B[j, k][mask2] + disVec) / (
                                                    C[j, k] + 1)  # update B with relative weight
                                        C[j, k] += 1

                    # if len(save_for_later_list) == 0:
                    #     break
                    if (num_vec_to_place == 0):
                        break
                    if (num_of_seq == 0):
                        break
            # b3 = datetime.datetime.now()
            # delta3 += (b3 - a3).total_seconds()
            mask1 = np.delete(mask1, mask1_placed_list, axis=0)  # update mask1
            mask2 = np.delete(mask2, mask2_placed_list, axis=0)  # update mask2

        # b = datetime.datetime.now()
        # print("delta1:    ", delta1)
        # print("delta2:    ", delta2)
        # print("delta3:    ", delta3)
        # print("total:    ", (b-a).total_seconds())

        return A

    def fasterfindOrderNP(self, x):  # fasterfindOrderNP is not faster then fastfindOrderNP
        delta1 = 0.0
        delta2 = 0.0
        delta3 = 0.0
        delta4 = 0.0
        DIM = int(np.sqrt(cfg.seq_in_samples))
        # A: contains the new arrangement, Initialized -1 to indicate square is empty
        A = np.zeros(shape=(DIM, DIM), dtype=int) - 1
        # B: contains the distance vectors, Initialized 100 to be far from chosen as minimum
        # B = np.zeros(shape=(DIM, DIM, cfg.seq_in_samples), dtype=float) + 100
        # C: Contains the number neighbors, Initialized 0 as there is no neighbors in the beginning
        C = np.zeros(shape=(DIM, DIM), dtype=int)

        small_B = np.array([np.zeros(cfg.seq_in_samples - 1)], np.int32)  # bug
        small_B = np.delete(small_B, 0, axis=0)  # bug

        # vec_in_small_B = 0
        # dic_mask1 = {}
        mask1 = np.array([np.zeros(2)], np.int32)  # bug  # holds vectors indices we should search in an argmin in B
        mask1 = np.delete(mask1, 0, axis=0)  # bug
        mask2 = np.arange(len(x[0]))  # holds vectors indices we didn't yet position in A

        # start with random vector
        # tempIndex = np.random.choice(vectorsIn)
        tempIndex = np.random.randint(0, len(x[0]))
        mask2 = np.delete(mask2, tempIndex, axis=0)
        # mask2 = mask2[mask2 != tempIndex]
        A[DIM // 2, DIM // 2] = tempIndex
        tempVec = x[0, tempIndex]
        disVec = np.mean(np.square(np.subtract(x[0][mask2], tempVec)),
                         axis=1)  # DistanceFromAllVectors (in mask2) # mabe to cut x, and no need for mask2???
        # disVec[tempIndex] = 100  # no need to look at this vector anywhere again
        for j in range(DIM // 2 - 1, DIM // 2 + 2):
            for k in range(DIM // 2 - 1, DIM // 2 + 2):
                if (0 <= j < DIM) & (0 <= k < DIM):
                    if (A[j, k] == -1):
                        # vec_in_small_B += 1
                        # dic_mask1[j, k] = vec_in_small_B
                        C[j, k] += 1
                        # B[j, k][mask2] = disVec
                        mask1 = np.concatenate((mask1, [[j, k]]), axis=0)
                        small_B = np.concatenate((small_B, [disVec]),
                                                 axis=0)  # maybe bug , add full vec then delete all

        # B[DIM // 2, DIM // 2] = 100  # no need to look at this square again
        # B[:, :, tempIndex] = 100  # no need to look at this vector anywhere again

        # go for all the rest vectors
        for i in range(len(x[0]) - 1):
            a1 = datetime.datetime.now()
            # smallB, delta = fastindexMask3D(B, mask1, mask2)
            # smallB, delta = indexMask3D(B, mask1, mask2)
            small_B_idx = np.unravel_index(np.argmin(small_B, axis=None),
                                           small_B.shape)  # square and vector with the minimum distance
            [idx_0, idx_1] = mask1[small_B_idx[0]]  # convert small_B_idx to real indeces in A/B/C (mask1)
            idx_2 = mask2[small_B_idx[1]]  # convert small_B_idx to real indeces in x (mask2)

            b1 = datetime.datetime.now()
            delta1 += (b1 - a1).total_seconds()
            a2 = datetime.datetime.now()

            # tempmask = np.ones(mask1.shape[0], dtype= bool)
            # tempmask[small_B_idx[0]] = False
            # small_B = small_B[list(tempmask.transpose())]

            mask1 = np.delete(mask1, small_B_idx[0], axis=0)  # remove chozen vec from mask1
            small_B = np.delete(small_B, small_B_idx[0], axis=0)  # remove chozen vec from small_B #2500-11
            mask2 = np.delete(mask2, small_B_idx[1], axis=0)  # remove chozen vec from mask2 #2500-0.1
            small_B = np.delete(small_B, small_B_idx[1],
                                axis=1)  # remove chozen vec from all vec in small_B ??? #2500-11

            # small_B = fastindexMask3D(small_B, mask1, mask2)

            A[idx_0, idx_1] = idx_2
            tempVec = x[0, idx_2]
            disVec = np.mean(np.square(np.subtract(x[0][mask2], tempVec)), axis=1)  # DistanceFromAllVectors (in mask2)
            # disVec = np.mean(np.square(np.subtract(x[0], tempVec)), axis=1)  # DistanceFromAllVectors

            b2 = datetime.datetime.now()
            delta2 += (b2 - a2).total_seconds()

            a3 = datetime.datetime.now()

            # update surrounding squares in B
            for j in range(idx_0 - 1, idx_0 + 2):
                for k in range(idx_1 - 1, idx_1 + 2):
                    # check if: (j , k) in matrix boundaries. (j , k) is not corrent vec / (j , k) is empty square in A
                    if (0 <= j < DIM) & (0 <= k < DIM):
                        if (A[j, k] == -1):
                            if (C[j, k] == 0):
                                mask1 = np.concatenate((mask1, [[j, k]]), axis=0)  # 2500-7
                                small_B = np.concatenate((small_B, [disVec]), axis=0)  # 2500-7
                            vec_to_update = np.where((mask1[:, 0] == j) & (mask1[:, 1] == k))[0][0]
                            small_B[vec_to_update] = (C[j, k] * small_B[vec_to_update] + disVec) / (
                                        C[j, k] + 1)  # update B with relative weight
                            C[j, k] += 1
                            # B[j, k][mask2] = (C[j, k] * B[j, k][mask2] + disVec) / (C[j, k] + 1)  # update B with relative weight

            b3 = datetime.datetime.now()
            delta3 += (b3 - a3).total_seconds()

            # B[index[0], index[1]] = 100  # no need to look at this square again
            # B[:, :, index[2]] = 100  # no need to look at this vector anywhere again
        a4 = datetime.datetime.now()
        b4 = datetime.datetime.now()
        delta4 += (b4 - a4).total_seconds()

        # print("counter:    ", counter)
        print("delta1:    ", delta1)
        print("delta2:    ", delta2)
        print("delta3:    ", delta3)
        print("delta4:    ", delta4)
        print("total:    ", delta1 + delta2 + delta3 + delta4)

        return A

    def generate_all_cubes(self, path):
        all_cubes = {}
        all_labels = {}
        for i, cube in enumerate(os.listdir(self.path)):
            if (cube != "temp"):
                all_cubes[i] = np.load(path + '/' + cube)
                if ("ill" in cube):
                    all_labels[i] = 1
                else:
                    all_labels[i] = 0
            else:
                continue

        print(len(all_cubes), ' reps')
        return all_cubes, all_labels

    def get_next_samples(self):
        while True:
            cubeIdx = random.randrange(len(self.all_cubes))

            randnums = np.random.randint(0, len(self.all_cubes[cubeIdx]), cfg.seq_in_samples)
            samples = self.all_cubes[cubeIdx][randnums]

            reorder_samples = self.reorder(samples)

            yield reorder_samples, self.all_labels[cubeIdx]


# dataset = tf.data.Dataset.from_generator(cubeGenerator.get_next_samples ,output_types=(tf.float64,tf.bool))
# dataset = dataset.batch(cfg.batch_size)
# dataset = dataset.prefetch(cfg.prefetch_batch_buffer)
# iter = tf.compat.v1.data.make_one_shot_iterator(dataset)

class Dataset(object):

    def __init__(self, path, gen):
        self.path = path
        self.generator = gen
        self.next_element = self.build_iterator(self.generator)

    def build_iterator(self, cube_gen: cubeGenerator):
        dataset = tf.data.Dataset.from_generator(cube_gen.get_next_samples, output_types=(tf.float32, tf.bool))
        dataset = dataset.batch(cfg.batch_size)
        dataset = dataset.prefetch(cfg.prefetch_batch_buffer)
        iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
        # iter = dataset.make_one_shot_iterator()
        element = iter.get_next()

        # return Inputs(element[0], element[1],)
        return dataset

###main###
# c = cubeGenerator(path="C:\\Users\\user\\Desktop\\limudim\\M.Sc\\RepertoiresClassification\\randomNsec\\Data\\ToyData\\ToyDataCleaned\\cubes")
# for i in c.get_next_samples():
#     print(i['isIll'])


# a = np.delete(a, np.argwhere(a == 10000))
# a = a[a != 10000]

# a3 = datetime.datetime.now()
# for i in range(300):
#     # n = np.where(b == i)[0]
#     b[i:10000] -= 1
# b3 = datetime.datetime.now()
# (b3 - a3).total_seconds()

# a3 = datetime.datetime.now()
# for i in range(300):
#     # n = np.where(b == i)[0]
#     np.unravel_index(np.argmin(b, axis=None), b.shape)
# b3 = datetime.datetime.now()
# n = (b3 - a3).total_seconds()
