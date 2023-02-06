
import numpy as np
import argparse
import os
from os.path import join
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, PReLU

# default_cleaned_path_cmv = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/Cleaned_32000_87_templates_folds'
default_cleaned_path_cmv = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/cleaned'
default_cleaned_path_simulated = r'/home/moshe/M.Sc/RepertoiresClassification/datasets/CMV/Cleaned'
default_cleaned_path_celiac = r'//home/moshe/M.Sc/RepertoiresClassification/datasets/celiac_data/Cleaned_32000_120nuc_merged'
default_cleaned_path_biomed = r'C:\Users\user\Desktop\limudim\M.Sc\RepertoiresClassification\datasets\BIOMED2\Cleaned_10000'
default_train_gan_path = r'C:\Users\user\Desktop\limudim\M.Sc\RepertoiresClassification\randomNsec\Data\celiac_data\XL\Cleaned\cubes\train'
default_train_path = r'C:\Users\user\Desktop\limudim\M.Sc\RepertoiresClassification\randomNsec\Data\celiac_data\XL\Cleaned\cubes\train'
default_validation_path = r'C:\Users\user\Desktop\limudim\M.Sc\RepertoiresClassification\randomNsec\Data\celiac_data\XL\Cleaned\cubes\validation'
default_arranged_cube_path = r"C:\Users\user\Desktop\limudim\M.Sc\RepertoiresClassification\randomNsec\Data\celiac_data\source//"
default_weights_path = r"C:\Users\user\Desktop\limudim\M.Sc\RepertoiresClassification\RepArrange\00\24_AE_weights.h5"
default_start_seq_path = r"C:\Users\user\Desktop\limudim\M.Sc\RepertoiresClassification\randomNsec\Data\celiac_data\start_seq_representative.npy"
class Configs(object):
    def __init__(self):

        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()

        # @ ----- model ----
        parser.add_argument('--model', type=str, default='CnnAutoencoder', help='model type')
        parser.add_argument('--model_type', type=int, default=21, help='model type under dataset type')
        parser.add_argument('--sub_model_type', type=int, default=3, help='sub model type under model type')

        # @ ----- control ----
        parser.add_argument('--mode', type=str, default='validation', help='train or validation')
        parser.add_argument('--device', type=str, default='/gpu:0', help='device to execute with')
        # parser.add_argument('--device', type=str, default='/cpu:0', help='device to execute with')
        parser.add_argument('--workers', type=int, default=16, help='load workers')
        parser.add_argument('--max_queue_size', type=int, default=20, help='max_queue_size = prefetch batch buffer')
        parser.add_argument('--prefetch_batch_buffer', type=int, default=1, help='prefetch batch buffer number')
        parser.add_argument('--shaffel_augmentation', type=bool, default=False, help='shaffel_augmentation')
        parser.add_argument('--use_embedding', type=bool, default=False, help='use_embedding')
        parser.add_argument('--use_tags', type=bool, default=False, help='use_tags')
        parser.add_argument('--random_sample', type=bool, default=False, help='random_sample or arranged samples')
        parser.add_argument('--load_weights', type=bool, default=False, help='load weights?')
        parser.add_argument('--freeze_weights', type=bool, default=False, help='freeze_weights')
        parser.add_argument('--load_points', type=bool, default=False, help='load points?')
        parser.add_argument('--balance_data', type=bool, default=False, help='balance_data for arrange_from_cube')
        parser.add_argument('--noised_sample', type=bool, default=False, help='add noise to sample')
        parser.add_argument('--AE', type=bool, default=False, help='autoencoder mode')
        parser.add_argument('--one_fold', type=bool, default=True, help='is all data in same path')
        parser.add_argument('--train_folds', type=str, nargs='+', default=['f2','f3','f4'], help='folds in train')
        parser.add_argument('--val_folds', type=str, nargs='+', default=['f0'], help='folds in validation')
        parser.add_argument('--test_folds', type=str, nargs='+', default=['f1'], help='folds in test')
        parser.add_argument('--extra_sample', type=int, default=1, help='extra_sample')
        parser.add_argument('--top_k', type=int, default=100, help='top_k')
        parser.add_argument('--seed', type=int, default=1, help='seed')
        parser.add_argument('--start_round', type=int, default=10, help='start_round for arrange_from_cube')
        parser.add_argument('--end_round', type=int, default=12, help='end_round for arrange_from_cube')
        parser.add_argument('--sleep_time', type=int, default=0, help='sleep_time for arrange_from_cube')
        parser.add_argument('--steps_per_epoch', type=int, default=2, help='train = steps_per_epoch, validation = steps_per_epoch // 10')
        parser.add_argument('--occlusion_size', type=int, default=316)
        parser.add_argument('--occlusion_stride', type=int, default=316)

        # @ ----------training ------
        parser.add_argument('--max_epoch', type=int, default=1, help='max epoch number')
        parser.add_argument('--batch_size', type=int, default=4, help='batch size number')
        parser.add_argument('--criterion', type=float, default=0.02, help='above criterion is ill, below is healthy')
        parser.add_argument('--temp_value', type=float, default=0.5, help='temp_value for any temp use')
        parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
        parser.add_argument('--decay', type=float, default=0.9, help='exponential decay rate')
        parser.add_argument('--initial_LR', type=float, default=0.001, help='initial learning rate')
        parser.add_argument('--activation_func', type=str, default='relu' , help='activation function')
        # parser.add_argument('--optimizer', type=str, default='Adam' , help='optimizer')
        parser.add_argument('--D_G_ratio', type=int, default=3 , help='discriminator generator ratio')


        # @ ----- Data ----
        parser.add_argument('--seq_in_samples', type=int, default=32000, help='number of sequences in each random choose from repertior') #32000
        parser.add_argument('--train_reps_num', type=int, default=100, help='number of repertoir for training')
        parser.add_argument('--val_reps_num', type=int, default=20, help='number of repertoir for validation')
        parser.add_argument('--rep_size', type=int, default=33000, help='mean repertoir size')

        # @ ----- data path ----
        parser.add_argument('--dataset', type=str, default='cmv', help='dataset type')
        parser.add_argument('--cleaned_path', type=str, default=default_cleaned_path_cmv, help='cleaned_path')
        parser.add_argument('--train_data_path', type=str, default=default_train_path, help='train_data_path')
        parser.add_argument('--val_data_path', type=str, default=default_validation_path, help='val_data_path')
        parser.add_argument('--train_data_gan_path', type=str, default=default_train_gan_path, help='default_train_gan_path')
        parser.add_argument('--arranged_cube_path', type=str, default=default_arranged_cube_path, help='arranged_cube_path')
        parser.add_argument('--weights_path', type=str, default=default_weights_path, help='weights path')
        parser.add_argument('--weights_name', type=str, default='21_3_4321_k100_f.h5', help='weights_name')
        parser.add_argument('--start_seq_path', type=str, default=default_start_seq_path, help='start_seq_path')
        parser.add_argument('--train_points', type=str, default='', help='train points path')
        parser.add_argument('--val_points', type=str, default='', help='val points path')
        parser.add_argument('--save_dir', type=str, default='00', help='directory to save logs and weights (01/02/03/04/05)')

        parser.add_argument('--data_path', type=str, default='/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/ordered_data/ordered_data_10000_1', help='from which directory to read the repertoires')

        ### divide the repertoires to train, validation and test (for each folder in ordered_data folder) ###
        folders_path = '/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/ordered_data'
        folders_list = os.listdir(folders_path)
        for f in range(len(folders_list)):
            files = os.listdir(folders_path + '/' + folders_list[f] + '/')
            f1, f2, f3, f4, f5 = np.array_split(files, 5)

        if train_folds == ['f2','f3','f4']:
            self.X_train = np.hstack((f2,f3,f4))
            self.X_val = f5
            self.X_test = f1
        elif train_folds == ['f1','f3','f4']:
            self.X_train = np.hstack((f1, f3, f4))
            self.X_val = f5
            self.X_test = f2
        elif train_folds == ['f1', 'f2', 'f4']:
            self.X_train = np.hstack((f1, f2, f4))
            self.X_val = f5
            self.X_test = f3
        elif train_folds == ['f1', 'f2', 'f5']:
            self.X_train = np.hstack((f1, f2, f5))
            self.X_val = f4
            self.X_test = f4
        elif train_folds == ['f1', 'f2', 'f3']:
            self.X_train = np.hstack((f1, f2, f3))
            self.X_val = f4
            self.X_test = f5
        #     X_train, X_test = train_test_split(files, test_size=0.2, shuffle=True)
        #     X_train, X_val = train_test_split(X_train, test_size=0.25)
        # self.X_train = X_train
        # self.X_val = X_val
        # self.X_test = X_test

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()
        # self.args.load_weights = True
        # self.args.load_points = True
        ## ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['test', 'shuffle']:
                exec('self.%s = self.args.%s' % (key, key))

        self.max_steps_in_train_epoch = int((self.args.train_reps_num * self.args.rep_size)/(self.args.seq_in_samples *self.args.batch_size)) + 4
        self.max_steps_in_val_epoch = int((self.args.val_reps_num * self.args.rep_size)/(self.args.seq_in_samples *self.args.batch_size))  + 4

        # @ ----- general ----
        self.dim = int(np.sqrt(self.args.seq_in_samples)) # 2dim

        # @ ------neural network-----
        self.LeakyReLU = LeakyReLU(alpha=0.3)
        self.PReLU = PReLU(shared_axes=[1])
        # @ ------optimizers-----
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.args.initial_LR,
            decay_steps=1000,
            decay_rate=self.args.decay)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) # lr_schedule is influenced by steps_per_epoch
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.initial_LR)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        # @ ------loss-----
        if((self.dataset == 'celiac') | (self.dataset == 'cmv')):
            self.loss = 'BinaryCrossentropy'
        elif (self.dataset == 'biomed'):
            self.loss = 'CategoricalCrossentropy'

        # self.optimizer = tf.keras.optimizers.Adam()
        # @ ----loss function-----
        # self.loss_func = tf.keras.losses.BinaryCrossentropy()
        # self.loss_func = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # self.loss_func = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # self.loss_func = tf.keras.losses.MeanSquaredError()

        # ------- name --------
        # self.train_data_name = 'cubes'

        # ---------- dir -------------
        # self.data_dir = join(self.dataset_dir, 'snli_1.0')

        # -------- path --------
        self.project_dir = os.getcwd()
        self.Data_dir = join(self.project_dir, 'Data')
        # self.save_dir = join(self.project_dir, self.save_dir)
        # self.weights_path = join(self.project_dir,  r'saved_weights/' + self.weights_name)

        # self.weights_path_save = join(self.save_dir,  self.dataset + r'_24_AE_weights_1.h5')
        self.weights_path = join(r'/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/results/'+ self.save_dir +self.weights_name)
        self.weights_path_save = join(r'/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/results/'+ self.save_dir +self.weights_name)

        self.train_points = join(self.project_dir, r'checkpoints\train_points.npy')
        self.val_points = join(self.project_dir, r'checkpoints\val_points.npy')
        # self.train_data_path = join(self.project_dir, 'randomNsec')
        # self.train_data_path = join(self.project_dir, 'Data/ToyData_Train/ToyDataCleaned/cubes')
        # self.val_data_path = join(self.project_dir, 'Data/ToyData_Val/ToyDataCleaned/cubes')
        #
        # self.train_data = "C:\\Users\\user\\Desktop\\limudim\\M.Sc\\RepertoiresClassification\\randomNsec\\Data\\ToyData_Train\\ToyDataCleaned\\cubes"
        # self.val_data = "C:\\Users\\user\\Desktop\\limudim\\M.Sc\\RepertoiresClassification\\randomNsec\\Data\\ToyData_Val\\ToyDataCleaned\\cubes"

        # self.train_data_path = join(self.data_dir, self.train_data_name)

        # dtype



cfg = Configs()