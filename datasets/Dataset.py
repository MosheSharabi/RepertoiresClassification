
import os
import random
import numpy as np
import tensorflow as tf
from configs import cfg


class Inputs(object):
    def __init__(self, samples, label):
        self.samples = samples
        self.label = label

class cubeGenerator(object):
    def __init__(self, path):
        self.path = path
        self.all_cubes , self.all_labels = self.generate_all_cubes(path)
        self.numOfSamples = 1000

    def generate_all_cubes(self, path):
        all_cubes = {}
        all_labels = {}
        for i, cube in enumerate(os.listdir(self.path)):
            all_cubes[i] = np.load(path + '/' + cube)
            if("ill" in cube):
                all_labels[i] = 1
            else:
                all_labels[i] = 0

        print(len(all_cubes) , ' reps')
        return  all_cubes ,all_labels

    def get_next_samples(self):
        while True:
            cubeIdx = random.randrange(len(self.all_cubes))

            randnums = np.random.randint(0, len(self.all_cubes[cubeIdx]), cfg.seq_in_samples)
            samples = self.all_cubes[cubeIdx][randnums]

            yield samples, self.all_labels[cubeIdx]


class Dataset(object):

    def __init__(self,path,gen):
        self.path = path
        self.generator = gen
        self.next_element = self.build_iterator(self.generator)

    def build_iterator(self, cube_gen: cubeGenerator):
        dataset = tf.data.Dataset.from_generator(cube_gen.get_next_samples ,output_types=(tf.float32,tf.bool))
        dataset = dataset.batch(cfg.batch_size)
        dataset = dataset.prefetch(cfg.prefetch_batch_buffer)
        iter = dataset.make_one_shot_iterator()
        element = iter.get_next()

        # return Inputs(element[0], element[1],)
        return iter


###main###
# c = cubeGenerator(path="C:\\Users\\user\\Desktop\\limudim\\M.Sc\\RepertoiresClassification\\randomNsec\\Data\\ToyData\\ToyDataCleaned\\cubes")
# for i in c.get_next_samples():
#     print(i['isIll'])