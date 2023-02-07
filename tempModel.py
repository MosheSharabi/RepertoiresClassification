import os
import logging
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
import numpy as np
from configs import cfg
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Conv3D, Conv1D, Dropout, BatchNormalization, \
    Conv2DTranspose, Conv3DTranspose, MaxPooling2D, MaxPooling3D, AveragePooling2D, Reshape, Add, Dropout, Lambda
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2

seed_value = cfg.seed
random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
tf.compat.v1.disable_eager_execution()


# ########  16  #######
# class CnnAutoencoder(Model):
#     def __init__(self):
#         super(CnnAutoencoder, self).__init__()
#         self.DIM = int(np.sqrt(cfg.seq_in_samples))
#         # ----ENCODER----
#         self.reshapeIN = Reshape([self.DIM, self.DIM, 84, 4])
#         self.conv1 = Conv3D(64, (1, 1, 3), strides=(1, 1, 3), activation=cfg.activation_func)  # 84 => 28
#         self.conv2 = Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # 28 => 28 => pooling = > 14
#         self.conv3 = Conv3D(32, (1, 1, 2), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # => 14 => 14 => pooling => 7
#         self.conv4 = Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # 7 => 7
#         self.poolingEncode = MaxPooling3D(strides=(1, 1, 2), padding='same')
#         self.reshapeFW = Reshape([self.DIM, self.DIM, 7 * 16])
#         self.d1 = Dense(16, activation='softmax')
#         # ----DECODER----
#         self.reshape1 = Reshape([self.DIM, self.DIM, 16, 1])
#         self.conv5 = Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # 16 => 16
#         self.t_conv1 = Conv3DTranspose(64, (1, 1, 3), strides=(1, 1, 2), activation='relu', padding='same')  # 16 => 32
#         self.conv6 = Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # 32 => 32
#         self.t_conv2 = Conv3DTranspose(64, (1, 1, 3), strides=(1, 1, 2), activation='relu', padding='same')  # 32 => 64
#         self.conv7 = Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # 64 => 64
#         self.t_conv3 = Conv3DTranspose(64, (1, 1, 3), strides=(1, 1, 2), activation='relu', padding='same')  # 64 => 128
#         self.conv8 = Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func)  # 128 => 126
#         self.t_conv4 = Conv3DTranspose(32, (1, 1, 3), strides=(1, 1, 2), activation='relu', padding='same')  # 126 => 252
#         self.decoded = Conv3D(4, (1, 1, 3), strides=(1, 1, 3), activation='sigmoid')  # 252 => 84
#
#         self.BN1 = BatchNormalization()
#         self.BN2 = BatchNormalization()
#         self.BN3 = BatchNormalization()
#
#         self.dropout = Dropout(cfg.dropout, noise_shape=None, seed=None)
#
#     def type(self):
#         print("CnnAutoencoder")
#
#     def call(self, x):
#         # ----ENCODER----
#         print("x:                   ", np.shape(x))
#         x = self.conv1(x)
#         print("conv1:               ", np.shape(x))
#         x = self.conv2(x)
#         x = self.poolingEncode(x)
#         print("conv2:               ", np.shape(x))
#         x = self.conv3(x)
#         x = self.poolingEncode(x)
#         print("conv3:               ", np.shape(x))
#         x = self.conv4(x)
#         print("conv4:               ", np.shape(x))
#         x = self.reshapeFW(x)
#         print("reshapeFW:           ", np.shape(x))
#         x = self.d1(x)
#         print("d1:                  ", np.shape(x))
#
#         # ----DECODER----
#
#         x = self.reshape1(x)
#         print("reshape1:            ", np.shape(x))
#         x = self.conv5(x)
#         print("conv5:               ", np.shape(x))
#         x = self.t_conv1(x)
#         x = self.BN1(x)
#         print("t_conv1:             ", np.shape(x))
#         x = self.conv6(x)
#         print("conv6:               ", np.shape(x))
#         x = self.t_conv2(x)
#         x = self.BN2(x)
#         print("t_conv2:             ", np.shape(x))
#         x = self.conv7(x)
#         print("conv7:               ", np.shape(x))
#         x = self.t_conv3(x)
#         x = self.BN3(x)
#         print("t_conv3:             ", np.shape(x))
#         x = self.conv8(x)
#         print("conv8:               ", np.shape(x))
#         x = self.t_conv4(x)
#         print("t_conv4:             ", np.shape(x))
#         x = self.decoded(x)
#         print("decoded:             ", np.shape(x))
#
#         return x


####### 24  #######
class CnnAutoencoder(Model):
    def __init__(self):
        super(CnnAutoencoder, self).__init__()
        self.dim = cfg.dim
        # ----ENCODER----
        self.conv1 = Conv3D(32, (1, 1, 3), strides=(1, 1, 3), activation=cfg.LeakyReLU, name='E_conv1')
        self.conv2 = Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv2')
        self.conv3 = Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3')
        self.conv4 = Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4')

        # ----DECODER----
        self.reshape = Reshape([self.dim, self.dim, 24, 1], name='D_reshape')
        self.conv5 = Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='D_conv5')
        self.t_conv1 = Conv3DTranspose(64, (1, 1, 3), strides=(1, 1, 2), activation='relu', padding='same',
                                       name='D_Tconv1')
        self.conv6 = Conv3D(64, (1, 1, 4), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='D_conv6')
        self.t_conv2 = Conv3DTranspose(64, (1, 1, 3), strides=(1, 1, 2), activation='relu', padding='same',
                                       name='D_Tconv2')
        self.conv7 = Conv3D(64, (1, 1, 5), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='D_conv7')
        self.t_conv3 = Conv3DTranspose(64, (1, 1, 3), strides=(1, 1, 2), activation='relu', padding='same',
                                       name='D_Tconv3')
        self.conv8 = Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='D_conv8')
        self.conv9 = Conv3D(4, (1, 1, 3), strides=(1, 1, 2), activation='sigmoid', name='D_conv9')

        self.BN1 = BatchNormalization(name='D_bn1')
        self.BN2 = BatchNormalization(name='D_bn2')
        self.BN3 = BatchNormalization(name='D_bn3')

    def type(self):
        print("CnnAutoencoder")

    def call(self, x):
        # ----ENCODER----
        # print("x:                   ", np.shape(x))
        x = self.conv1(x)
        # print("conv1:               ", np.shape(x))
        x = self.conv2(x)
        # print("conv2:               ", np.shape(x))
        x = self.conv3(x)
        # print("conv3:               ", np.shape(x))
        x = self.conv4(x)
        # print("conv4:               ", np.shape(x))
        x = tf.squeeze(x, axis=4)
        # print("squeeze:             ", np.shape(x))
        # ----DECODER----

        x = self.reshape(x)
        # print("reshape:             ", np.shape(x))
        x = self.conv5(x)
        # print("conv5:               ", np.shape(x))
        x = self.t_conv1(x)
        x = self.BN1(x)
        # print("t_conv1:             ", np.shape(x))
        x = self.conv6(x)
        # print("conv6:               ", np.shape(x))
        x = self.t_conv2(x)
        x = self.BN2(x)
        # print("t_conv2:             ", np.shape(x))
        x = self.conv7(x)
        # print("conv7:               ", np.shape(x))
        x = self.t_conv3(x)
        x = self.BN3(x)
        # print("t_conv3:             ", np.shape(x))
        x = self.conv8(x)
        # print("conv8:               ", np.shape(x))
        x = self.conv9(x)
        # print("conv9:               ", np.shape(x))

        return x


####### 24  #######
class CnnDecoder(Model):
    def __init__(self):
        super(CnnDecoder, self).__init__()
        self.dim = cfg.dim
        # ----DECODER----
        self.reshape = Reshape([self.dim, self.dim, 24, 1], name='D_reshape')
        self.conv7 = Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='D_conv7')
        self.t_conv1 = Conv3DTranspose(128, (1, 1, 3), strides=(1, 1, 2), activation='relu', padding='same',
                                       name='D_Tconv1')
        self.conv8 = Conv3D(64, (1, 1, 4), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='D_conv8')
        self.t_conv2 = Conv3DTranspose(128, (1, 1, 3), strides=(1, 1, 2), activation='relu', padding='same',
                                       name='D_Tconv2')
        self.conv9 = Conv3D(64, (1, 1, 5), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='D_conv9')
        self.t_conv3 = Conv3DTranspose(64, (1, 1, 3), strides=(1, 1, 2), activation='relu', padding='same',
                                       name='D_Tconv3')
        self.conv10 = Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='D_conv10')
        self.conv11 = Conv3D(4, (1, 1, 3), strides=(1, 1, 2), activation='sigmoid', name='D_conv11')

        self.BN1 = BatchNormalization(name='D_bn1')
        self.BN2 = BatchNormalization(name='D_bn2')
        self.BN3 = BatchNormalization(name='D_bn3')

    def type(self):
        print("CnnDecoder")

    def call(self, x):
        # ----DECODER----

        x = self.reshape(x)
        # print("reshape:             ", np.shape(x))
        x = self.conv7(x)
        # print("conv7:               ", np.shape(x))
        x = self.t_conv1(x)
        x = self.BN1(x)
        # print("t_conv1:             ", np.shape(x))
        x = self.conv8(x)
        # print("conv8:               ", np.shape(x))
        x = self.t_conv2(x)
        x = self.BN2(x)
        # print("t_conv2:             ", np.shape(x))
        x = self.conv9(x)
        # print("conv9:               ", np.shape(x))
        x = self.t_conv3(x)
        x = self.BN3(x)
        # print("t_conv3:             ", np.shape(x))
        x = self.conv10(x)
        # print("conv10:               ", np.shape(x))
        x = self.conv11(x)
        # print("conv11:               ", np.shape(x))
        return x


# ####### 16  #######
# class CnnEncoder(Model):
#     def __init__(self):
#         super(CnnEncoder, self).__init__()
#         self.DIM = int(np.sqrt(cfg.seq_in_samples))
#         # ----ENCODER----
#         self.reshapeIN = Reshape([self.DIM, self.DIM, 84, 4])
#         self.conv1 = Conv3D(64, (1, 1, 3), strides=(1, 1, 3), activation=cfg.activation_func)  # 84 => 28
#         self.conv2 = Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # 28 => 28 => pooling = > 14
#         self.conv3 = Conv3D(32, (1, 1, 2), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # => 14 => 14 => pooling => 7
#         self.conv4 = Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # 7 => 7
#         self.poolingEncode = MaxPooling3D(strides=(1, 1, 2), padding='same')
#         self.reshapeHW = Reshape([1, cfg.seq_in_samples, 7 * 16])
#         # self.reshapeFW = Reshape([self.DIM, self.DIM, 7 * 16])
#         self.d1 = Dense(16, activation='softmax')
#
#     def type(self):
#         print("CnnEncoder")
#
#     def call(self, x):
#         # ----ENCODER----
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.poolingEncode(x)
#         x = self.conv3(x)
#         x = self.poolingEncode(x)
#         x = self.conv4(x)
#         x = self.reshapeHW(x)
#         x = self.d1(x)
#         return x

####### 24  #######
class CnnEncoder(Model):
    def __init__(self):
        super(CnnEncoder, self).__init__()
        self.dim = cfg.dim
        # ----ENCODER----
        self.conv1 = Conv3D(32, (1, 1, 3), strides=(1, 1, 3), activation=cfg.LeakyReLU, name='E_conv1')
        self.conv2 = Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv2')
        self.conv3 = Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3')
        self.conv4 = Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4')

    def type(self):
        print("CnnEncoder")

    def call(self, x):
        # ----ENCODER----
        # print("x:                   ", np.shape(x))
        x = self.conv1(x)
        # print("conv1:               ", np.shape(x))
        x = self.conv2(x)
        # print("conv2:               ", np.shape(x))
        x = self.conv3(x)
        # print("conv3:               ", np.shape(x))
        x = self.conv4(x)
        # print("conv4:               ", np.shape(x))
        x = tf.squeeze(x, axis=4)
        # print("squeeze:             ", np.shape(x))
        return x


class Classifier(Model):
    def __init__(self):
        super(Classifier, self).__init__()
        # self._dynamic = True
        self.DIM = cfg.dim

        # ----ENCODER----
        self.conv1 = Conv3D(64, (1, 1, 3), strides=(1, 1, 3), activation=cfg.activation_func)  # 84 => 28
        self.conv2 = Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func,
                            padding='same')  # 28 => 28 => pooling = > 14
        self.conv3 = Conv3D(32, (1, 1, 2), strides=(1, 1, 1), activation=cfg.activation_func,
                            padding='same')  # => 14 => 14 => pooling => 7
        self.conv4 = Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # 7 => 7

        self.poolingEncode = MaxPooling3D(strides=(1, 1, 2), padding='same')

        # self.reshape1 = Reshape([cfg.seq_in_samples, -1])
        self.reshapeHW = Reshape([1, cfg.seq_in_samples, 7 * 16])
        self.reshapeFW = Reshape([self.dim, self.dim, 7 * 16])

        self.d1 = Dense(16, activation='softmax')

        # ----Reorder----
        # self.reorder = tf.gather_nd()

        # ----Classifier----
        self.conv5 = Conv2D(64, (2, 2), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
        self.conv6 = Conv2D(64, (3, 3), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
        self.conv7 = Conv2D(32, (3, 3), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
        self.conv8 = Conv2D(16, (3, 3), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
        self.conv9 = Conv2D(8, (3, 3), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
        self.conv10 = Conv2D(4, (2, 2), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
        self.conv11 = Conv2D(1, (2, 2), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM

        self.poolingClassifier = MaxPooling2D(strides=(2, 2), padding='same')

        self.dropout = Dropout(cfg.dropout, noise_shape=None, seed=None)

        self.d2 = Dense(1, activation='softmax')

        self.flatten = Flatten()

        self.BN1 = BatchNormalization()
        self.BN2 = BatchNormalization()
        self.BN3 = BatchNormalization()

    def type(self):
        print("Classifier")

    def DistanceFromAllVectors(self, x, vectorsIn, vec):
        dis = np.mean(np.square(np.subtract(x, vec)), axis=1)
        return dis

    @tf.function
    def findOrderTF(self, x):
        DIM = int(np.sqrt(cfg.seq_in_samples))
        print(" DIM  ", DIM)
        vectorsIn = np.arange(x.get_shape()[1])
        # A: contains the new arrangement, Initialized -1 to indicate square is empty
        A = np.zeros(shape=(DIM, DIM), dtype=int) - 1
        # B: contains the distance vectors, Initialized 100 to be far from chosen as minimum
        B = np.zeros(shape=(DIM, DIM, cfg.seq_in_samples), dtype=float) + 100
        # C: Contains the number neighbors, Initialized 0 as there is no neighbors in the beginning
        C = np.zeros(shape=(DIM, DIM), dtype=int)

        # start with random vector
        tempIndex = np.random.choice(vectorsIn)
        tempVec = x[:, tempIndex]
        print("tempVec ", tempVec.get_shape())
        print("x ", x.get_shape())
        disVec = tf.reduce_mean(tf.square(x - tempVec), axis=1)  # DistanceFromAllVectors
        print("disVec ", np.shape(disVec))

        disVec[tempIndex].add = 100  # no need to look at this vector anywhere again
        A[DIM / 2, DIM / 2, 0] = tempIndex
        for j in range(DIM / 2 - 1, DIM / 2 + 1):
            for k in range(DIM / 2 - 1, DIM / 2 + 1):
                if (A[j, k] == -1):
                    B[j, k] = disVec
                    C[j, k] += 1

        # go for all vectors
        for i in range(len(x)):
            index = np.unravel_index(np.argmin(B, axis=None), B.shape)  # square and vector with the minimum distance
            A[index[0], index[1], 0] = index[2]
            tempVec = x[index[2]]
            disVec = tf.reduce_mean(tf.square(x - tempVec), axis=1)  # DistanceFromAllVectors

            # update surrounding squares in B
            for j in range(index[0] - 1, index[0] + 1):
                for k in range(index[1] - 1, index[1] + 1):
                    # check if: (j , k) in matrix boundaries. (j , k) is not corrent vec / (j , k) is empty square in A
                    if (0 <= j <= DIM) & (0 <= k <= DIM) & (A[j, k] == -1):
                        B[j, k] = (C[j, k] * B[j, k] + disVec) / (C[j, k] + 1)  # update B with relative weight
                        C[j, k] += 1

            B[index[0], index[1]] = 100  # no need to look at this square again
            B[:, :, index[2]] = 100  # no need to look at this vector anywhere again

        return A

    def findOrderNP(self, x):
        DIM = int(np.sqrt(cfg.seq_in_samples))
        vectorsIn = np.arange(len(x))
        # A: contains the new arrangement, Initialized -1 to indicate square is empty
        A = np.zeros(shape=(DIM, DIM), dtype=int) - 1
        # B: contains the distance vectors, Initialized 100 to be far from chosen as minimum
        B = np.zeros(shape=(DIM, DIM, cfg.seq_in_samples), dtype=float) + 100
        # C: Contains the number neighbors, Initialized 0 as there is no neighbors in the beginning
        C = np.zeros(shape=(DIM, DIM), dtype=int)

        # start with random vector
        tempIndex = np.random.choice(vectorsIn)
        tempVec = x[tempIndex]
        disVec = np.mean(np.square(np.subtract(x, tempVec)), axis=1)  # DistanceFromAllVectors
        disVec[tempIndex] = 100  # no need to look at this vector anywhere again
        A[DIM / 2, DIM / 2, 0] = tempIndex
        for j in range(DIM / 2 - 1, DIM / 2 + 1):
            for k in range(DIM / 2 - 1, DIM / 2 + 1):
                if (A[j, k] == -1):
                    B[j, k] = disVec
                    C[j, k] += 1

        # go for all vectors
        for i in range(len(x)):
            index = np.unravel_index(np.argmin(B, axis=None), B.shape)  # square and vector with the minimum distance
            A[index[0], index[1], 0] = index[2]
            tempVec = x[index[2]]
            disVec = np.mean(np.square(np.subtract(x, tempVec)), axis=1)  # DistanceFromAllVectors

            # update surrounding squares in B
            for j in range(index[0] - 1, index[0] + 1):
                for k in range(index[1] - 1, index[1] + 1):
                    # check if: (j , k) in matrix boundaries. (j , k) is not corrent vec / (j , k) is empty square in A
                    if (0 <= j <= DIM) & (0 <= k <= DIM) & (A[j, k] == -1):
                        B[j, k] = (C[j, k] * B[j, k] + disVec) / (C[j, k] + 1)  # update B with relative weight
                        C[j, k] += 1

            B[index[0], index[1]] = 100  # no need to look at this square again
            B[:, :, index[2]] = 100  # no need to look at this vector anywhere again

        return A

    # @tf.function
    def half_way(self, x):
        # tf.print("x:    ", x)
        # print(type(x))
        # print(type(x.numpy()))
        # print(type(x.eval(session=tf.compat.v1.Session())))

        # ----ENCODER----
        # print(np.shape(x))
        x = self.conv1(x)
        # print(np.shape(x))
        x = self.conv2(x)
        # print(np.shape(x))
        x = self.poolingEncode(x)
        # print(np.shape(x))
        x = self.conv3(x)
        # print(np.shape(x))
        x = self.poolingEncode(x)
        # print(np.shape(x))
        x = self.conv4(x)
        # tf.print("x:    ", x)
        # print(x.get_shape())
        x = self.reshapeHW(x)
        # print(x.get_shape())
        x = self.d1(x)
        # print(x.get_shape())

        return x

    def full_way(self, x):
        # ----ENCODER----
        # print("x:                   ", np.shape(x))
        x = self.conv1(x)
        # print("conv1:               ", np.shape(x))
        x = self.conv2(x)
        x = self.poolingEncode(x)
        # print("conv2:               ", np.shape(x))
        x = self.conv3(x)
        x = self.poolingEncode(x)
        # print("conv3:               ", np.shape(x))
        x = self.conv4(x)
        # print("conv4:               ", np.shape(x))
        x = self.reshapeFW(x)
        # print("reshapeFW:           ", np.shape(x))
        x = self.d1(x)
        # print("d1:                  ", np.shape(x))

        # ----Reorder----
        # order = self.findOrderTF(x)
        # order = self.findOrder(x)
        # order = self.findOrder(tf.identity(x))
        # x = self.reorder(x,order)
        # x = tf.gather_nd(x,order)
        # print(np.shape(x))

        # ----Classifier----
        x = self.conv5(x)
        x = self.poolingClassifier(x)
        # print("conv5:               ", np.shape(x))
        x = self.conv6(x)
        x = self.poolingClassifier(x)
        # print("conv6:               ", np.shape(x))
        x = self.conv7(x)
        x = self.poolingClassifier(x)
        # print("conv7:               ", np.shape(x))
        x = self.conv8(x)
        x = self.poolingClassifier(x)
        # print("conv8:               ", np.shape(x))
        # x = self.flatten(x)
        # print("flatten:             ", np.shape(x))
        # x = self.d2(x)
        # print("d2:                  ", np.shape(x))
        x = self.conv9(x)
        x = self.poolingClassifier(x)
        # print("conv9:               ", np.shape(x))
        x = self.conv10(x)
        x = self.poolingClassifier(x)
        # print("conv10:               ", np.shape(x))
        x = self.conv11(x)
        x = self.poolingClassifier(x)
        # print("conv11:               ", np.shape(x))
        x = self.flatten(x)
        # print("flatten:             ", np.shape(x))

        return x

    def call(self, x, isFullWay):
        if (isFullWay == True):
            return self.full_way(x)
        elif (isFullWay == False):
            return self.half_way(x)


# ########  16  #######
# class Classifier_new(Model):
#     def __init__(self):
#         super(Classifier_new, self).__init__()
#         # self._dynamic = True
#         self.DIM = int(np.sqrt(cfg.seq_in_samples))
#
#         # ----ENCODER----
#         self.conv1 = Conv3D(64, (1, 1, 3), strides=(1, 1, 3), activation=cfg.activation_func)  # 84 => 28
#         self.conv2 = Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # 28 => 28 => pooling = > 14
#         self.conv3 = Conv3D(32, (1, 1, 2), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # => 14 => 14 => pooling => 7
#         self.conv4 = Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.activation_func, padding='same')  # 7 => 7
#
#         self.poolingEncode = MaxPooling3D(strides=(1, 1, 2), padding='same')
#
#         # self.reshape1 = Reshape([cfg.seq_in_samples, -1])
#         self.reshapeHW = Reshape([1, cfg.seq_in_samples, 7 * 16])
#         self.reshapeFW = Reshape([self.DIM, self.DIM, 7 * 16])
#
#         self.d1 = Dense(16, activation='softmax')
#
#         # ----Reorder----
#         # self.reorder = tf.gather_nd()
#
#         # ----Classifier----
#         self.conv5 = Conv2D(64, (2, 2), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
#         self.conv6 = Conv2D(64, (3, 3), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
#         self.conv7 = Conv2D(32, (3, 3), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
#         self.conv8 = Conv2D(16, (3, 3), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
#         self.conv9 = Conv2D(8, (3, 3), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
#         self.conv10 = Conv2D(4, (2, 2), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
#         self.conv11 = Conv2D(1, (2, 2), strides=(1, 1), activation=cfg.activation_func, padding='same')  # DIM => DIM
#
#
#         self.poolingClassifier = MaxPooling2D(strides=(2, 2), padding='same')
#
#         self.dropout = Dropout(cfg.dropout, noise_shape=None, seed=None)
#
#         self.d2 = Dense(1, activation='softmax')
#
#         self.flatten = Flatten()
#
#     def type(self):
#         print("Classifier_new")
#
#     def call(self, x):
#         # ----ENCODER----
#         # ----ENCODER----
#         # print("x:                   ", np.shape(x))
#         x = self.conv1(x)
#         # print("conv1:               ", np.shape(x))
#         x = self.conv2(x)
#         x = self.poolingEncode(x)
#         # print("conv2:               ", np.shape(x))
#         x = self.conv3(x)
#         x = self.poolingEncode(x)
#         # print("conv3:               ", np.shape(x))
#         x = self.conv4(x)
#         # print("conv4:               ", np.shape(x))
#         x = self.reshapeFW(x)
#         # print("reshapeFW:           ", np.shape(x))
#         x = self.d1(x)
#         # print("d1:                  ", np.shape(x))
#
#
#         # ----Classifier----
#         x = self.conv5(x)
#         x = self.poolingClassifier(x)
#         # print("conv5:               ", np.shape(x))
#         x = self.conv6(x)
#         x = self.poolingClassifier(x)
#         # print("conv6:               ", np.shape(x))
#         x = self.conv7(x)
#         x = self.poolingClassifier(x)
#         # print("conv7:               ", np.shape(x))
#         x = self.conv8(x)
#         x = self.poolingClassifier(x)
#         # print("conv8:               ", np.shape(x))
#         # x = self.flatten(x)
#         # print("flatten:             ", np.shape(x))
#         # x = self.d2(x)
#         # print("d2:                  ", np.shape(x))
#         x = self.conv9(x)
#         x = self.poolingClassifier(x)
#         # print("conv9:               ", np.shape(x))
#         x = self.conv10(x)
#         x = self.poolingClassifier(x)
#         # print("conv10:               ", np.shape(x))
#         x = self.conv11(x)
#         x = self.poolingClassifier(x)
#         # print("conv11:               ", np.shape(x))
#         x = self.flatten(x)
#         # print("flatten:             ", np.shape(x))
#
#         return x


# ########  24  #######
class Classifier_new(Model):
    def __init__(self, dataset='cmv'):
        super(Classifier_new, self).__init__()
        self.dim = cfg.dim
        # ----ENCODER----
        self.conv1 = Conv3D(32, (1, 1, 3), strides=(1, 1, 3), activation=cfg.LeakyReLU, name='E_conv1')
        self.conv2 = Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv2')
        self.conv3 = Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3')
        self.conv4 = Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4')

        # ----Classifier----
        self.conv5 = Conv2D(16, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv5')
        self.conv6 = Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv6')
        self.conv7 = Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv7')
        self.conv8 = Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv8')
        if ((cfg.dataset == 'celiac') | (cfg.dataset == 'cmv')):
            self.d2 = Dense(1, activation='sigmoid')
        elif (cfg.dataset == 'biomed'):
            self.d2 = Dense(3, activation='softmax')

        self.dropout = Dropout(cfg.dropout, noise_shape=None, seed=None)
        self.flatten = Flatten()

    def type(self):
        print("Classifier_new")

    def call(self, x):
        # ----ENCODER----
        # print("x:                   ", np.shape(x))
        x = self.conv1(x)
        # x = self.dropout(x)
        # print("conv1:               ", np.shape(x))
        x = self.conv2(x)
        # x = self.dropout(x)
        # print("conv2:               ", np.shape(x))
        x = self.conv3(x)
        # x = self.dropout(x)
        # print("conv3:               ", np.shape(x))
        x = self.conv4(x)
        # x = self.dropout(x)
        # print("conv4:               ", np.shape(x))
        x = tf.squeeze(x, axis=4)
        # print("squeeze:             ", np.shape(x))

        # ----Classifier----
        x = self.conv5(x)
        x = self.dropout(x)
        # x = self.poolingClassifier(x)
        # print("conv5:               ", np.shape(x))
        x = self.conv6(x)
        x = self.dropout(x)
        # x = self.poolingClassifier(x)
        # print("conv6:               ", np.shape(x))
        x = self.conv7(x)
        x = self.dropout(x)
        # x = self.poolingClassifier(x)
        # print("conv7:               ", np.shape(x))
        x = self.conv8(x)
        x = self.dropout(x)
        # x = self.poolingClassifier(x)
        print("conv8:               ", np.shape(x))
        x = self.flatten(x)
        print("flatten:             ", np.shape(x))
        x = self.d2(x)

        return x


def SequentialClassifier():
    if (cfg.dataset == 'celiac'):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 84, 4),
                                   activation=cfg.LeakyReLU, name='E_conv1'),
            tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv2'),
            tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
            tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
            tf.keras.layers.Reshape((cfg.dim, cfg.dim, 24)),
            tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv5'),
            tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv6'),
            tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv7'),
            tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv8'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    if (cfg.dataset == 'cmv'):
        return cmv_models()
    elif (cfg.dataset == 'biomed'):
        return biomed_models()


def biomed_models():
    if (cfg.model_type == 1):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 84, 4),
                                   activation=cfg.LeakyReLU, name='E_conv1'),
            tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv2'),
            tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Reshape((cfg.dim, cfg.dim, 24)),
            tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv5'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv6'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv7'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv8'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        return model
    elif (cfg.model_type == 2):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 84, 4),
                                   activation=cfg.LeakyReLU, name='E_conv1'),
            tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv2'),
            tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv21'),
            tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv22'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Reshape((cfg.dim, cfg.dim, 24)),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='C_conv51'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='C_conv52'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv5'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv6'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv7'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv8'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        return model
    elif (cfg.model_type == 3):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 1), input_shape=(cfg.dim, cfg.dim, 84, 4),
                                   activation=cfg.LeakyReLU, name='E_conv1'),
            tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv2'),
            tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Reshape((cfg.dim, cfg.dim, 76)),
            tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv5'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv6'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv7'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv8'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        return model

    elif (cfg.model_type == 4): # arrange by length

        # ##### 1 ##### with top_k
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 96, 4)),
        #     tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv1'),
        #     tf.keras.layers.Conv2D(16, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
        #     tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
        #     tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
        #     tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 21)),
        #
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv1'),
        #     # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
        #     tf.keras.layers.MaxPooling1D(pool_size=32),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv2'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv3'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv4'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv5'),
        #     tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k}),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv6'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
        # ], name="4_1")

        ##### 2 ##### stride encoder and clasifier
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 96, 4)),
            tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1'),

            tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
                                   name='E_conv2'),
            tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
            tf.keras.layers.Conv2D(16, (1, 4), strides=(1, 2), activation=cfg.LeakyReLU,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
            tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
            tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
            tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 10)),

            tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv1'),
            # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
            tf.keras.layers.MaxPooling1D(pool_size=32),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(32, 3, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv2'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(32, 3, strides=2, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv3'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(32, 3, strides=2, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv4'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(32, 3, strides=2, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv5'),
            tf.keras.layers.Conv1D(16, 3, strides=2, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv6'),
            tf.keras.layers.Conv1D(8, 3, strides=2, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv7'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))

            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
            #                       bias_regularizer=l2(0.00001)),
            # tf.keras.layers.Dropout(rate=cfg.dropout),
            # tf.keras.layers.Dense(16, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
            #                       bias_regularizer=l2(0.00001)),
            # tf.keras.layers.Dropout(rate=cfg.dropout),
            # tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
        ], name="4_2")
        return model


def cmv_models():
    if (cfg.model_type == 1):   # 2D models
        if (cfg.sub_model_type == 1):   # most naive 2D model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv5'),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv6'),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv7'),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv8'),
                tf.keras.layers.Dropout(rate=0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

        if (cfg.sub_model_type == 2):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv21'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv22'),
                tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='C_conv51'),
                tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='C_conv52'),
                tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv5'),
                tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv6'),
                tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv7'),
                tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv8'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

        if (cfg.sub_model_type == 3):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv21'),
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv22'),
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv23'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv24'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv25'),
                tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='C_conv51'),
                tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='C_conv52'),
                tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='C_conv53'),
                tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='C_conv54'),
                tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='C_conv55'),
                tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv5'),
                tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv6'),
                tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv7'),
                tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv8'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        if (cfg.sub_model_type == 4): # resnet 2D
            def ResNet(input_shape=(cfg.dim, cfg.dim, 87, 4)):
                # Define the input as a tensor with shape input_shape
                X_input = Input(input_shape)

                # Stage 1
                x = Conv3D(64, (1, 1, 3), strides=(1, 1, 3), activation=cfg.LeakyReLU, name='E_conv1')(X_input)
                # Stage 2
                x = conv_block_3D(x, stage=2, block='a', filters=[16, 16, 16], conv_kernel=(1, 1, 4))
                x = identity_block_3D(x, stage=2, block='b', filters=[16, 16, 16], conv_kernel=(1, 1, 4))
                x = identity_block_3D(x, stage=2, block='c', filters=[16, 16, 16], conv_kernel=(1, 1, 4))
                x = conv_block_3D(x, stage=2, block='d', filters=[16, 8, 1], conv_kernel=(1, 1, 4))
                x = Reshape((cfg.dim, cfg.dim, 29))(x)
                # Stage 3
                x = conv_block_2D(x, stage=3, block='a', filters=[32, 32, 16], conv_kernel=(1, 1))
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(rate=0.3)(x)
                x = identity_block_2D(x, stage=3, block='b', filters=[16, 16, 16], conv_kernel=(1, 1))
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(rate=0.3)(x)
                x = identity_block_2D(x, stage=3, block='c', filters=[16, 16, 16], conv_kernel=(1, 1))
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(rate=0.3)(x)
                # Stage 4 - for 22500
                x = identity_block_2D(x, stage=4, block='a', filters=[16, 16, 16], conv_kernel=(1, 1))
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(rate=0.3)(x)
                x = identity_block_2D(x, stage=4, block='b', filters=[16, 16, 16], conv_kernel=(1, 1))
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(rate=0.3)(x)
                x = conv_block_2D(x, stage=4, block='c', filters=[16, 16, 4], conv_kernel=(1, 1))
                x = MaxPooling2D(pool_size=(2, 2))(x)
                # Stage 5
                x = Flatten()(x)
                x = Dense(1, activation='sigmoid')(x)
                # Create model
                model = Model(inputs=X_input, outputs=x, name='ResNet')
                return model

            model = ResNet()

        if (cfg.sub_model_type == 5): # 2D with naive global pooling
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv2'),
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv5'),
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv6'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv7'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv8'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv9'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 13)),
                tf.keras.layers.Dense(1, activation='sigmoid'),
                tf.keras.layers.AveragePooling2D(pool_size=(cfg.dim, cfg.dim), strides=None),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        if (cfg.sub_model_type == 6):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51'),
                tf.keras.layers.Dropout(rate=0.3),
                tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv5'),
                tf.keras.layers.Dropout(rate=0.3),
                tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv6'),
                tf.keras.layers.Dropout(rate=0.3),
                tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv7'),
                tf.keras.layers.Dropout(rate=0.3),
                tf.keras.layers.Conv2D(1, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv8'),
                tf.keras.layers.Dropout(rate=0.2),  # check if to remove
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        if (cfg.sub_model_type == 7):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(16, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv5'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(8, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv6'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv7'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv8'),
                # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  # add to 22500
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
        if (cfg.sub_model_type == 8): # 2D top-k
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(cfg.dim, cfg.dim, 87, 4)),
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv1'),
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 2), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 2), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 2), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                # tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 14)),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 10)),

                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.MaxPooling2D(pool_size=(6,6)),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.Reshape(((cfg.dim//6) * (cfg.dim//6), 32)),
                top_k_pixel_pooling_class(k=cfg.top_k, pool_size=0, is_sorted=False),
                # tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k}),
                # top_k_pixel_pooling_class(k=10, is_sorted=False, input_dim=cfg.seq_in_samples // 32, pool_size=100),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001))
            ], name="21_3")

        return model

    if (cfg.model_type == 82):
        if (cfg.sub_model_type == 1):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same', name='E_conv2'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(16, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="82")

        if (cfg.sub_model_type == 2):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv2'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv3'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv4'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv5'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv6'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv7'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv8'),
                tf.keras.layers.Conv3D(1, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv9'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 29)),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.001),
                                       bias_regularizer=l2(0.001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.001),
                                       bias_regularizer=l2(0.001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.001),
                                       bias_regularizer=l2(0.001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.001),
                                       bias_regularizer=l2(0.001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(16, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.001),
                                       bias_regularizer=l2(0.001), name='C_conv9'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.001),
                                       bias_regularizer=l2(0.001), name='C_conv10'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))
            ], name="82")

        if (cfg.sub_model_type == 3): # broad end, 3 dense
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(16, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(32, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="82")

        if (cfg.sub_model_type == 4):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv2'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv3'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv4'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv5'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv6'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv7'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv8'),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv9'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv10'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv11'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv12'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv13'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv14'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv15'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv16'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv17'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv18'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv19'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv20'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv21'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv22'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv23'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv24'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv25'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(8, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv26'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Conv3D(1, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv27'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 29)),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.0001),
                                       bias_regularizer=l2(0.0001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.0001),
                                       bias_regularizer=l2(0.0001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.0001),
                                       bias_regularizer=l2(0.0001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(16, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.0001),
                                       bias_regularizer=l2(0.0001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(8, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.0001),
                                       bias_regularizer=l2(0.0001), name='C_conv9'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.0001),
                                       bias_regularizer=l2(0.0001), name='C_conv10'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(32, activation='sigmoid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))
            ], name="82")

        if (cfg.sub_model_type == 5):   # pooling kernels
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="82")

        if (cfg.sub_model_type == 6):   # encoder cnn kernels
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 5), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same', name='E_conv2'),
                tf.keras.layers.Conv3D(8, (1, 1, 5), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv3D(1, (1, 1, 5), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 21)),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(16, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="82")

        if (cfg.sub_model_type == 7):  # 82_5 with dilation
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4), activation=cfg.LeakyReLU, name='E_conv1'),

                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same', name='E_conv2'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 2), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 4), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 11)),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="82")

        if (cfg.sub_model_type == 8):   #like 82_7 with 8 filter in E_conv7
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 84, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),

                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 2),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 4),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 80)),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="82")

        if (cfg.sub_model_type == 9):   # no encoder
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 3),
                                       input_shape=(cfg.dim * cfg.extra_sample, cfg.dim, 87, 4), activation=cfg.LeakyReLU,
                                       name='E_conv1'),

                tf.keras.layers.Reshape((cfg.dim * cfg.extra_sample, cfg.dim, 29 * 32)),
                tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv1'),
                tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv2'),
                tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv3'),
                tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv4'),

                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="82")

        return model

    if (cfg.model_type == 9):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 84, 4),
                                   activation=cfg.LeakyReLU, name='E_conv1'),
            tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv2'),
            tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv3'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv4'),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Reshape((cfg.dim, cfg.dim, 28)),
            tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2D(16, (2, 2), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='C_conv5'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2D(8, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv6'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv7'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv8'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ], name="9")
        return model
    if (cfg.model_type == 10):  # PointNet basic

        input_sample = tf.keras.layers.Input(shape=(cfg.dim, cfg.dim, 84, 4))
        e = tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 3), activation=cfg.LeakyReLU, name='E_conv1')(
            input_sample)
        e = tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv2')(e)
        e = tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv3')(e)
        e = tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv4')(e)
        e = tf.keras.layers.Reshape((cfg.dim, cfg.dim, 28))(e)
        # check
        e = tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51')(e)
        e = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(e)
        # Point Net
        x = tf.keras.layers.Convolution2D(64, 1, activation='relu')(e)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Convolution2D(128, 1, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Convolution2D(1024, 1, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        global_feature = tf.keras.layers.MaxPooling2D(pool_size=cfg.dim // 2)(x)
        c = tf.keras.layers.Dense(512, activation='relu')(global_feature)
        c = tf.keras.layers.BatchNormalization()(c)
        c = tf.keras.layers.Dense(256, activation='relu')(c)
        c = tf.keras.layers.BatchNormalization()(c)
        c = tf.keras.layers.Dropout(rate=0.7)(c)
        c = tf.keras.layers.Dense(1, activation='sigmoid')(c)
        prediction = Flatten()(c)
        model = Model(inputs=input_sample, outputs=prediction)
        return model
    if (cfg.model_type == 11):  # PointNet basic

        # encoder
        input_sample = tf.keras.layers.Input(shape=(cfg.dim, cfg.dim, 84, 4))
        e = tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 3), activation=cfg.LeakyReLU, name='E_conv1')(
            input_sample)
        e = tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv2')(e)
        e = tf.keras.layers.Conv3D(4, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv3')(e)
        e = tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv4')(e)
        e = tf.keras.layers.Reshape((cfg.seq_in_samples, 28))(e)

        # input_Transformation_net
        x = tf.keras.layers.Convolution1D(32, 1, activation='relu', input_shape=(cfg.seq_in_samples, 28))(e)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Convolution1D(32, 1, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Convolution1D(128, 1, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=cfg.seq_in_samples)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(28 * 28, weights=[np.zeros([128, 28 * 28]), np.eye(28).flatten().astype(np.float32)])(
            x)
        input_T = tf.keras.layers.Reshape((28, 28))(x)

        # forward net
        g = tf.keras.layers.Lambda(mat_mul, arguments={'B': input_T})(e)
        g = tf.keras.layers.Convolution1D(32, 1, input_shape=(cfg.seq_in_samples, 28), activation='relu')(g)
        g = tf.keras.layers.BatchNormalization()(g)
        g = tf.keras.layers.Convolution1D(32, 1, input_shape=(cfg.seq_in_samples, 28), activation='relu')(g)
        g = tf.keras.layers.BatchNormalization()(g)

        # feature transform net
        f = tf.keras.layers.Convolution1D(32, 1, activation='relu')(g)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.Convolution1D(32, 1, activation='relu')(f)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.Convolution1D(128, 1, activation='relu')(f)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.MaxPooling1D(pool_size=cfg.seq_in_samples)(f)
        f = tf.keras.layers.Dense(128, activation='relu')(f)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.Dense(128, activation='relu')(f)
        f = tf.keras.layers.BatchNormalization()(f)
        f = tf.keras.layers.Dense(32 * 32, weights=[np.zeros([128, 32 * 32]), np.eye(32).flatten().astype(np.float32)])(
            f)
        feature_T = Reshape((32, 32))(f)

        # forward net
        g = tf.keras.layers.Lambda(mat_mul, arguments={'B': feature_T})(g)
        g = tf.keras.layers.Convolution1D(32, 1, activation='relu')(g)
        g = tf.keras.layers.BatchNormalization()(g)
        g = tf.keras.layers.Convolution1D(32, 1, activation='relu')(g)
        g = tf.keras.layers.BatchNormalization()(g)
        g = tf.keras.layers.Convolution1D(128, 1, activation='relu')(g)
        g = tf.keras.layers.BatchNormalization()(g)

        # global_feature
        global_feature = tf.keras.layers.MaxPooling1D(pool_size=cfg.seq_in_samples)(g)

        c = tf.keras.layers.Dense(64, activation='relu')(global_feature)
        c = tf.keras.layers.BatchNormalization()(c)
        c = tf.keras.layers.Dense(32, activation='relu')(c)
        c = tf.keras.layers.BatchNormalization()(c)
        c = tf.keras.layers.Dropout(rate=0.7)(c)
        c = tf.keras.layers.Dense(1, activation='sigmoid')(c)
        prediction = Flatten()(c)
        model = Model(inputs=input_sample, outputs=prediction)
        model._name = "11"
        return model
    if (cfg.model_type == 12):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 84, 4),
                                   activation=cfg.LeakyReLU, name='E_conv1'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv3D(64, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv2'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv3D(64, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv3'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv3D(128, (1, 1, 2), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv4'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.MaxPooling3D(pool_size=(1, 1, 28)),
            tf.keras.layers.Reshape((cfg.dim, cfg.dim, 128)),
            tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv52'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(64, (2, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv5'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(64, (2, 2), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv6'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv7'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv8'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(1, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, name='C_conv9'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ], name="12")
        return model
    if (cfg.model_type == 13):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 84, 4),
                                   activation=cfg.LeakyReLU, name='E_conv1'),
            tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv2'),
            tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
            tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
            tf.keras.layers.Reshape((cfg.dim, cfg.dim, 24)),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51'),
            tf.keras.layers.Lambda(max_pixel_pooling),  # add sum
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv5'),
            tf.keras.layers.Lambda(max_pixel_pooling),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv6'),
            tf.keras.layers.Lambda(max_pixel_pooling),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2D(32, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv7'),
            tf.keras.layers.Lambda(max_pixel_pooling),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2D(16, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv8'),
            tf.keras.layers.Lambda(max_pixel_pooling),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv2D(4, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv9'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ], name="13")
        return model
    if (cfg.model_type == 14):
        if (cfg.sub_model_type == 1):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Reshape(((cfg.dim // 2)**2, 128)),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 1024}),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 128}),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 32}),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(16, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv9'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ], name="14")

        if (cfg.sub_model_type == 2):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51'),
                tf.keras.layers.Reshape(((cfg.dim) ** 2, 64)),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 4096}),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv5'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 1024}),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv6'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 256}),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv7'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 64}),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(16, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv9'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ], name="14")

        if (cfg.sub_model_type == 3):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51'),
                tf.keras.layers.Reshape(((cfg.dim) ** 2, 128)),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 4096}),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(128, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv5'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 1024}),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv6'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 256}),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv7'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 100}),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ], name="14")

        if (cfg.sub_model_type == 4):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 25)),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51'),
                tf.keras.layers.Reshape(((cfg.dim) ** 2, 128)),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(128, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv7'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 100}),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ], name="14")

        if (cfg.sub_model_type == 5):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same', name='E_conv2'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same', name='E_conv3'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same', name='E_conv4'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same', name='E_conv5'),
                tf.keras.layers.Reshape(((cfg.dim) ** 2, 28)),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 100}),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="14")

        if (cfg.sub_model_type == 6):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv3'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv4'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv5'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 29)),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Reshape(((cfg.dim // 2) ** 2, 64)),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 100}),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.BatchNormalization(renorm=True),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="14")

        if (cfg.sub_model_type == 7):   # like 6 different pooling kernel
            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                       activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv3'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv4'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv5'),
                tf.keras.layers.Reshape((cfg.dim, cfg.dim, 29)),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, name='C_conv51'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Reshape(((cfg.dim // 4) ** 2, 64)),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 100}),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="14")

        if (cfg.sub_model_type == 8):   # 14_7 classifaier, with 82_8 encoder(diliation)

            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim * cfg.extra_sample, cfg.dim, 87, 4), activation=cfg.LeakyReLU, name='E_conv1'),

                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same', name='E_conv2'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 2), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 4), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
                tf.keras.layers.Reshape((cfg.dim * cfg.extra_sample, cfg.dim, 11)),

                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Reshape((-1, 64)),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k}),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=0.3),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="14")

        if (cfg.sub_model_type == 9):   # 9 no encoder

            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(32, (1, 1, 3), strides=(1, 1, 3),
                                       input_shape=(cfg.dim * cfg.extra_sample, cfg.dim, 87, 4), activation=cfg.LeakyReLU,
                                       name='E_conv1'),

                tf.keras.layers.Reshape((cfg.dim * cfg.extra_sample, cfg.dim, 29*32)),
                tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv1'),
                tf.keras.layers.Conv2D(256, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv2'),
                tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv3'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_12'),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv4'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_11'),
                tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Reshape((-1, 64)),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k}),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=0.3),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="14")

        if (cfg.sub_model_type == 10):   # top_k with global pooling

            model = tf.keras.Sequential([
                tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3),
                                       input_shape=(cfg.dim * cfg.extra_sample, cfg.dim, 87, 4), activation=cfg.LeakyReLU,
                                       name='E_conv1'),

                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
                                       name='E_conv2'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 2),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 4),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
                tf.keras.layers.Reshape((cfg.dim * cfg.extra_sample * cfg.dim, 11)),

                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(128, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(256, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(512, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k}),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Conv1D(512, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.AveragePooling1D(pool_size=(cfg.top_k)),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=0.2),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="14")

        return model
    if (cfg.model_type == 15):
        # 1
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Reshape((cfg.dim * cfg.dim,  84, 4), input_shape=(cfg.dim, cfg.dim, 84, 4)),
        #     tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1'),
        #     tf.keras.layers.Conv2D(32, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv2'),
        #     tf.keras.layers.Conv2D(32, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
        #     tf.keras.layers.Conv2D(32, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
        #     tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
        #     tf.keras.layers.BatchNormalization(renorm=True),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv2D(32, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv5'),
        #     tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
        #     tf.keras.layers.BatchNormalization(renorm=True),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv6'),
        #     tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
        #     tf.keras.layers.BatchNormalization(renorm=True),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv2D(1, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv7'),
        #     tf.keras.layers.Reshape((cfg.dim * cfg.dim, 1)),
        #     tf.keras.layers.Lambda(sum),
        #     tf.keras.layers.Dense(1, activation='sigmoid')
        # ], name="15")

        # # 2
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Reshape((cfg.dim * cfg.dim, 84, 4), input_shape=(cfg.dim, cfg.dim, 84, 4)),
        #     tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1'),
        #     tf.keras.layers.Conv2D(32, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv2'),
        #     tf.keras.layers.Conv2D(32, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
        #     tf.keras.layers.Conv2D(32, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
        #     tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
        #     tf.keras.layers.BatchNormalization(renorm=True),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv2D(32, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), name='E_conv5'),
        #     tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
        #     tf.keras.layers.BatchNormalization(renorm=True),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), name='E_conv6'),
        #     tf.keras.layers.MaxPooling2D(pool_size=(1, 2)),
        #     tf.keras.layers.BatchNormalization(renorm=True),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv2D(1, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), name='E_conv7'),
        #     tf.keras.layers.Reshape((cfg.dim * cfg.dim, 1)),
        #     tf.keras.layers.Lambda(sum),
        #     tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))
        # ], name="15")

        # 3
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((cfg.dim * cfg.dim, 84, 4), input_shape=(cfg.dim, cfg.dim, 84, 4)),
            tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1'),
            tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv2'),
            tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv3'),
            tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv4'),
            tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv5'),
            tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv6'),
            tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv7'),
            tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv8'),
            tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv9'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv10'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv11'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv12'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv13'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv14'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv15'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv16'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv17'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv18'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv19'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv20'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv21'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv22'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv23'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv24'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv25'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv26'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(8, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv27'),
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(1, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv28'),
            tf.keras.layers.Reshape((cfg.dim * cfg.dim, 1)),
            tf.keras.layers.Lambda(sum),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ], name="15")
        return model
    if (cfg.model_type == 16):
        # #########   1   #########
        # input_shape = (cfg.dim, cfg.dim, 84, 4)
        # X_input = Input(input_shape)
        # # encoder
        # e = tf.keras.layers.Reshape((cfg.dim * cfg.dim, 84, 4), input_shape=input_shape)(X_input)
        # e = tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1')(e)
        # e = tf.keras.layers.Conv2D(32, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv2')(e)
        # e = tf.keras.layers.Conv2D(32, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv3')(e)
        # e = tf.keras.layers.Conv2D(32, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv4')(e)
        #
        # # weight net
        # w = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), name='w_MP1')(e)
        # w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN1')(w)
        # w = tf.keras.layers.Dropout(rate=0.2)(w)
        # w = tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                        bias_regularizer=l2(0.00001), name='w_conv1')(w)
        # w = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), name='w_MP2')(w)
        # w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN2')(w)
        # w = tf.keras.layers.Dropout(rate=0.2)(w)
        # w = tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                        bias_regularizer=l2(0.00001), name='w_conv2')(w)
        # w = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), name='w_MP3')(w)
        # w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN3')(w)
        # w = tf.keras.layers.Dropout(rate=0.2)(w)
        # w = tf.keras.layers.Conv2D(1, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                        bias_regularizer=l2(0.00001), name='w_conv3')(w)
        # w = tf.keras.layers.Reshape((cfg.dim * cfg.dim, 1))(w)
        #
        # # forward net
        # g = tf.keras.layers.Lambda(top_k_atention_pooling, arguments={'w': w,'k': 100}, name='g_Lambda')(e)
        #
        # g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN1')(g)
        # g = tf.keras.layers.Dropout(rate=0.2)(g)
        # g = tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                        bias_regularizer=l2(0.00001), name='g_conv1')(g)
        # g = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), name='g_MP1')(g)
        # g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN2')(g)
        # g = tf.keras.layers.Dropout(rate=0.2)(g)
        # g = tf.keras.layers.Conv2D(16, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                        bias_regularizer=l2(0.00001), name='g_conv2')(g)
        # g = tf.keras.layers.MaxPooling2D(pool_size=(1, 2), name='g_MP2')(g)
        # g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN3')(g)
        # g = tf.keras.layers.Dropout(rate=0.2)(g)
        # g = tf.keras.layers.Conv2D(1, (1, 2), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='g_conv3')(g)
        # g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN4')(g)
        # g = tf.keras.layers.Dropout(rate=0.2)(g)
        # g = tf.keras.layers.Flatten()(g)
        # g = tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                       bias_regularizer=l2(0.00001))(g)
        # g = tf.keras.layers.Dropout(rate=0.2)(g)
        # g = tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                       bias_regularizer=l2(0.00001))(g)
        # g = tf.keras.layers.Dropout(rate=0.2)(g)
        # g = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(g)
        #
        #
        # s = tf.keras.layers.Lambda(sum, name='s_Lambda')(w)
        # s = tf.keras.layers.Dense(1, activation='sigmoid')(s)
        #
        # out = Add()([s, g])
        # # Create model
        # model = Model(inputs=X_input, outputs=out, name='16')

        # #########   2   #########
        # input_shape = (cfg.dim, cfg.dim, 84, 4)
        # X_input = Input(input_shape)
        # # encoder
        # e = tf.keras.layers.Reshape((cfg.dim * cfg.dim, 84, 4), input_shape=input_shape)(X_input)
        # e = tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1')(e)
        #
        # e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same', name='E_conv2')(e)
        # e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3')(e)
        # e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 2), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4')(e)
        # e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 4), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5')(e)
        # e = tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6')(e)
        # e = tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1), kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7')(e)
        # e = tf.keras.layers.Reshape((cfg.dim * cfg.dim, 10), input_shape=input_shape)(e)
        #
        # # weight net
        # w = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='w_conv1')(e)
        # w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN1')(w)
        # w = tf.keras.layers.Dropout(rate=cfg.dropout, name='w_DO1')(w)
        # w = tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='w_conv2')(w)
        # w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN2')(w)
        # w = tf.keras.layers.Dropout(rate=cfg.dropout, name='w_DO2')(w)
        # w = tf.keras.layers.Conv1D(16, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='w_conv3')(w)
        # w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN3')(w)
        # w = tf.keras.layers.Dropout(rate=cfg.dropout, name='w_DO3')(w)
        # w = tf.keras.layers.Conv1D(8, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='w_conv4')(w)
        # w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN4')(w)
        # w = tf.keras.layers.Dropout(rate=cfg.dropout, name='w_DO4')(w)
        # w = tf.keras.layers.Conv1D(1, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='w_conv5')(w)
        # w = tf.keras.layers.Reshape((cfg.dim * cfg.dim, 1))(w)
        #
        # # forward net
        # x = tf.keras.layers.Multiply()([e, w])
        #
        # g = tf.keras.layers.Lambda(top_k_atention_pooling, arguments={'w': w, 'k': 100}, name='g_Lambda')(x)
        # # g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN1')(g)
        # # g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO1')(g)
        # g = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='g_conv1')(g)
        # g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN2')(g)
        # g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO2')(g)
        # g = tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='g_conv2')(g)
        # g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN3')(g)
        # g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO3')(g)
        # g = tf.keras.layers.Conv1D(16, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='g_conv3')(g)
        # g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN4')(g)
        # g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO4')(g)
        # g = tf.keras.layers.Conv1D(8, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='g_conv4')(g)
        # g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN5')(g)
        # g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO5')(g)
        # g = tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='g_conv5')(g)
        #
        # # dense net
        # d = tf.keras.layers.Flatten()(g)
        # d = tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(d)
        # d = tf.keras.layers.Dropout(rate=cfg.dropout, name='d_DO1')(d)
        # d = tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(d)
        # d = tf.keras.layers.Dropout(rate=cfg.dropout, name='d_DO2')(d)
        # d = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(d)
        #
        # model = Model(inputs=X_input, outputs=d, name='16')

        #########   3   #########
        input_shape = (cfg.dim, cfg.dim, 84, 4)
        X_input = Input(input_shape)
        # encoder
        e = tf.keras.layers.Reshape((cfg.dim * cfg.dim, 84, 4), input_shape=input_shape)(X_input)
        e = tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1')(e)

        e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
                                   name='E_conv2')(e)
        e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3')(e)
        e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 2),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4')(e)
        e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 4),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5')(e)
        e = tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6')(e)
        e = tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7')(e)
        e = tf.keras.layers.Reshape((cfg.dim * cfg.dim, 10), input_shape=input_shape)(e)

        # forward net

        g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN1')(e)
        g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO1')(g)
        g = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='g_conv1')(g)
        g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN2')(g)
        g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO2')(g)
        g = tf.keras.layers.Conv1D(128, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='g_conv2')(g)
        g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN3')(g)
        g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO3')(g)
        g = tf.keras.layers.Conv1D(256, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='g_conv3')(g)
        g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN4')(g)
        g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO4')(g)
        g = tf.keras.layers.Conv1D(512, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='g_conv4')(g)

        # weight net
        w = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='w_conv1')(g)
        w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN1')(w)
        w = tf.keras.layers.Dropout(rate=cfg.dropout, name='w_DO1')(w)
        w = tf.keras.layers.Conv1D(16, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='w_conv2')(w)
        w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN2')(w)
        w = tf.keras.layers.Dropout(rate=cfg.dropout, name='w_DO2')(w)
        w = tf.keras.layers.Conv1D(1, 1, strides=1, activation='sigmoid', kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='w_conv3')(w)
        w = tf.keras.layers.Reshape((cfg.dim * cfg.dim, 1))(w)

        # top_k_atention_pooling_1
        l = tf.keras.layers.Lambda(top_k_atention_pooling_1, arguments={'w': w, 'k': cfg.top_k}, name='g_Lambda')(g)

        # forward net
        g1 = tf.keras.layers.BatchNormalization(renorm=True, name='g1_BN5')(l)
        g1 = tf.keras.layers.Dropout(rate=cfg.dropout, name='g1_DO5')(g1)
        g1 = tf.keras.layers.Conv1D(512, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                    bias_regularizer=l2(0.00001), name='g1_conv5')(g1)

        # global k pooling
        g1 = tf.keras.layers.AveragePooling1D(pool_size=(cfg.top_k))(g1)

        # dense net
        d = tf.keras.layers.Flatten()(g1)
        d = tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001))(d)
        g1 = tf.keras.layers.BatchNormalization(renorm=True, name='d_BN1')(l)
        d = tf.keras.layers.Dropout(rate=cfg.dropout, name='d_DO1')(d)
        d = tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001))(d)
        d = tf.keras.layers.Dropout(rate=cfg.dropout, name='d_DO2')(d)
        d = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001))(d)

        model = Model(inputs=X_input, outputs=d, name='16')

        return model

    if (cfg.model_type == 17):  # like 14_8 with tags

        # 1 - like 14_8 with output tags
        model = tf.keras.Sequential([
            tf.keras.layers.Conv3D(64, (1, 1, 3), strides=(1, 1, 3), input_shape=(cfg.dim, cfg.dim, 87, 4),
                                   activation=cfg.LeakyReLU, name='E_conv1'),

            tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
                                   name='E_conv2'),
            tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
            tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 2),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
            tf.keras.layers.Conv3D(16, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 4),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
            tf.keras.layers.Conv3D(8, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
            tf.keras.layers.Conv3D(1, (1, 1, 3), strides=(1, 1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
            tf.keras.layers.Reshape((cfg.dim, cfg.dim, 11)),

            tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv5'),
            # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Reshape(((cfg.dim // 4) ** 2, 64)),
            tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv6'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv7'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv8'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv9'),
            tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': 100}),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv10'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001)),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001)),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(2, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                  name='d_2')
        ], name="17")

        return model
    if (cfg.model_type == 18):  # attention

        #########   1   #########
        input_shape = (cfg.dim, cfg.dim, 84, 4)
        X_input = Input(input_shape)
        # encoder
        e = tf.keras.layers.Reshape((cfg.dim * cfg.dim, 84, 4), input_shape=input_shape)(X_input)
        e = tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1')(e)

        e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
                                   name='E_conv2')(e)
        e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3')(e)
        e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 2),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4')(e)
        e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 4),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5')(e)
        e = tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6')(e)
        e = tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7')(e)
        e = tf.keras.layers.Reshape((cfg.dim * cfg.dim, 10), input_shape=input_shape)(e)

        # attention net
        w = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='w_conv1')(e)
        w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN1')(w)
        w = tf.keras.layers.Dropout(rate=cfg.dropout, name='w_DO1')(w)
        w = tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='w_conv2')(w)
        w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN2')(w)
        w = tf.keras.layers.Dropout(rate=cfg.dropout, name='w_DO2')(w)
        w = tf.keras.layers.Conv1D(16, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='w_conv3')(w)
        w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN3')(w)
        w = tf.keras.layers.Dropout(rate=cfg.dropout, name='w_DO3')(w)
        w = tf.keras.layers.Conv1D(8, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='w_conv4')(w)
        w = tf.keras.layers.BatchNormalization(renorm=True, name='w_BN4')(w)
        w = tf.keras.layers.Dropout(rate=cfg.dropout, name='w_DO4')(w)
        w = tf.keras.layers.Conv1D(1, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='w_conv5')(w)
        w = tf.keras.layers.Reshape((cfg.dim * cfg.dim, 1))(w)
        w = tf.keras.layers.Softmax(axis=0)(w)

        # forward net

        g = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='g_conv1')(e)
        g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN2')(g)
        g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO2')(g)
        g = tf.keras.layers.Conv1D(128, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='g_conv2')(g)
        g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN3')(g)
        g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO3')(g)
        g = tf.keras.layers.Conv1D(256, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='g_conv3')(g)
        g = tf.keras.layers.BatchNormalization(renorm=True, name='g_BN4')(g)
        g = tf.keras.layers.Dropout(rate=cfg.dropout, name='g_DO4')(g)
        g = tf.keras.layers.Conv1D(512, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='g_conv4')(g)

        # attention
        x = tf.keras.layers.Multiply()([g, w])
        x = tf.keras.layers.GlobalMaxPooling1D()(x)

        # dense net
        d = tf.keras.layers.Flatten()(x)
        d = tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001))(d)
        d = tf.keras.layers.Dropout(rate=cfg.dropout, name='d_DO1')(d)
        d = tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001))(d)
        d = tf.keras.layers.Dropout(rate=cfg.dropout, name='d_DO2')(d)
        d = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001))(d)

        model = Model(inputs=X_input, outputs=d, name='18')
        return model

    if (cfg.model_type == 19):  # arrange by length

        # ##### 1 #####
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=(cfg.seq_in_samples, 84, 4)),
        #     tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1'),
        #
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
        #                            name='E_conv2'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 2),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 4),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
        #     tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
        #     tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
        #     tf.keras.layers.Reshape((cfg.seq_in_samples, 10)),
        #
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv5'),
        #     # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
        #     tf.keras.layers.MaxPooling1D(pool_size=200),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv6'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv7'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv8'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv9'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
        #     tf.keras.layers.Dropout(rate=0.3),
        #     tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                            bias_regularizer=l2(0.00001), name='C_conv10'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
        # ], name="19")

        ##### 2 ##### with top_k
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(cfg.seq_in_samples, 84, 4)),
            tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1'),

            tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
                                   name='E_conv2'),
            tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
            tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 2),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
            tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 4),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
            tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
            tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
            tf.keras.layers.Reshape((cfg.seq_in_samples, 10)),

            tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv5'),
            # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
            tf.keras.layers.MaxPooling1D(pool_size=32),
            tf.keras.layers.Dropout(rate=0.2),
            # tf.keras.layers.Reshape((-1, 64)),
            tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv6'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv7'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv8'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv9'),
            tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k}),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv10'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001)),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001)),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
        ], name="19")

        # ##### 3_1 ##### dilation classifier
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=(cfg.seq_in_samples, 84, 4)),
        #     tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1'),
        #
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
        #                            name='E_conv2'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 2),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 4),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
        #     tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
        #     tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
        #     tf.keras.layers.Reshape((cfg.seq_in_samples, 10)),
        #
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5'),
        #     # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
        #     tf.keras.layers.MaxPooling1D(pool_size=32),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv7'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=2,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv8'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=4,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv9'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
        #     tf.keras.layers.Dropout(rate=0.3),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=8,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv10'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=16,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv11'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_7'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=32,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv12'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_8'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=64,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv13'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_9'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(32, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=128,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv14'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_10'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(4, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=64,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv15'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_11'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
        # ], name="19")

        # ##### 3_2 ##### MaxPooling1D and dilation classifier and regular dense
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=(cfg.seq_in_samples, 84, 4)),
        #     tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1'),
        #
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
        #                            name='E_conv2'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 2),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 4),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
        #     tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
        #     tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
        #     tf.keras.layers.Reshape((cfg.seq_in_samples, 10)),
        #
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5'),
        #     # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
        #     tf.keras.layers.MaxPooling1D(pool_size=8),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv7'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=2,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv8'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=4,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv9'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
        #     tf.keras.layers.Dropout(rate=0.3),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=8,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv10'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=16,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv11'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_7'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=32,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv12'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_8'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=64,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv13'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_9'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=128,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv14'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_10'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=256,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv15'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_11'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(32, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=512,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv16'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_12'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(4, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=350,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv17'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_13'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
        # ], name="19")

        # ##### 4 ##### dilation classifier with dense on one point
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=(cfg.seq_in_samples, 84, 4)),
        #     tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 3), activation=cfg.LeakyReLU, name='E_conv1'),
        #
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
        #                            name='E_conv2'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 2),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 4),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
        #     tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
        #     tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
        #     tf.keras.layers.Reshape((cfg.seq_in_samples, 10)),
        #
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5'),
        #     # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv7'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=2,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv8'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=4,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv9'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
        #     tf.keras.layers.Dropout(rate=0.3),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=8,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv10'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=16,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv11'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_7'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=32,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv12'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_8'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=64,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv13'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_9'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=128,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv14'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_10'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=256,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv15'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_11'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=512,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv16'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_12'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=1024,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv17'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_13'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=2048,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv18'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_14'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(32, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=4096,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv19'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_15'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Conv1D(4, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=2048,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv20'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_16'),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=0.2),
        #     tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
        # ], name="19")

        return model

    if (cfg.model_type == 20):  # amino acid & arrange by length

        # # ##### 1 #####
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=(cfg.seq_in_samples, 27, 20)),
        #     tf.keras.layers.Conv2D(128, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv1'),
        #
        #     tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
        #                            name='E_conv2'),
        #     tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
        #     tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 2),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 4),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
        #     tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
        #     tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
        #     tf.keras.layers.Reshape((cfg.seq_in_samples, 13)),
        #
        #     tf.keras.layers.Conv1D(128, 1, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv1'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv2'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=2,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv3'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=4,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv4'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=8,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=16,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=32,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv7'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_7'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=64,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv8'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_8'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=128,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv9'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_9'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=256,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv10'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_10'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=512,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv11'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_11'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=1024,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv12'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_12'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(256, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=2048,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv13'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_13'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(256, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=4096,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv14'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_14'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(512, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=8192,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv15'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_15'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
        # ], name="20")


        # # ##### 2 ##### dilation classifier with regular dens
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=(cfg.seq_in_samples, 27, 20)),
        #     tf.keras.layers.Conv2D(128, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
        #                            name='E_conv1'),
        #
        #     tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
        #                            name='E_conv2'),
        #     tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
        #     tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 2),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
        #     tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 4),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
        #     tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
        #                            dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
        #     tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
        #                            dilation_rate=(1, 1),
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
        #     tf.keras.layers.Reshape((cfg.seq_in_samples, 13)),
        #
        #     tf.keras.layers.Conv1D(128, 1, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv1'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv2'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=2,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv3'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=4,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv4'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=8,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=16,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=32,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv7'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_7'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=64,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv8'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_8'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=128,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv9'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_9'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=256,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv10'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_10'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(128, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=512,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv11'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_11'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=1024,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv12'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_12'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(32, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=2048,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv13'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_13'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(16, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=4096,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv14'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_14'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Conv1D(4, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=8162,
        #                            kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv15'),
        #     tf.keras.layers.BatchNormalization(renorm=True, name='BN_15'),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
        #                           bias_regularizer=l2(0.00001)),
        #     tf.keras.layers.Dropout(rate=cfg.dropout),
        #     tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
        # ], name="20")

        # ##### 3 ##### stride classifier with regular dens
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(cfg.seq_in_samples, 27, 20)),
            tf.keras.layers.Conv2D(128, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                   name='E_conv1'),

            tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), padding='same',
                                   name='E_conv2'),
            tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
            tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 2),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
            tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 4),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
            tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                   dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
            tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                   dilation_rate=(1, 1),
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv7'),
            tf.keras.layers.Reshape((cfg.seq_in_samples, 13)),

            tf.keras.layers.Conv1D(256, 1, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv1'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(256, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv2'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(256, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv3'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(256, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv4'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(256, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(256, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(256, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv7'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_7'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(128, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv8'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_8'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(64, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv9'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_9'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(32, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv10'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_10'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Conv1D(4, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv11'),
            tf.keras.layers.BatchNormalization(renorm=True, name='BN_11'),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001)),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001)),
            tf.keras.layers.Dropout(rate=cfg.dropout),
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
        ], name="20")

        return model

    if (cfg.model_type == 21):  # 32767_87 & arrange by length
        if (cfg.sub_model_type == 1):
            ##### 1 ##### stride encoder - stride classifier - regular dens
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(cfg.seq_in_samples, 87, 4)),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv1'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                tf.keras.layers.Reshape((cfg.seq_in_samples, 10)),

                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv1'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 3, strides=1, activation=cfg.LeakyReLU, dilation_rate=1,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv2'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv3'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv4'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_7'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(32, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_8'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(16, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_9'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(8, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_10'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(4, 3, strides=2, activation=cfg.LeakyReLU, dilation_rate=1,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv11'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_11'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="21_1")

            class residual_class(tf.keras.Model): #one dense layer
                def __init__(self, hidden):
                    super(top_k_pixel_pooling_class, self).__init__()
                    self.w1 = tf.keras.layers.Dense(hidden, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001))

                def call(self, x):
                    x_shortcut = x
                    x = self.w1(x)
                    x = Add()([x, x_shortcut])
                    x = cfg.LeakyReLU(x)
                    return x

        if (cfg.sub_model_type == 2):
            ##### 2 ##### mlp encoder - top k - regular dens
            model = tf.keras.Sequential([
                ### encoder ###
                tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 87, 4)),
                tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 348)),
                tf.keras.layers.Dense(256, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='E_dense1'),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='E_dense2'),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='E_dense3'),
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='E_dense4'),
                tf.keras.layers.Dense(10, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='E_dense5'),
                ### classifier ###
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.MaxPooling1D(pool_size=32),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),
                top_k_pixel_pooling_class(k=cfg.top_k, pool_size=0, is_sorted=False),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001))
            ], name="21_2")

        #### 3 ##### stride encoder - top k - regular dens
        if (cfg.sub_model_type == 3):
            model = tf.keras.Sequential([
                # tf.keras.layers.Input(shape=(50, 50, 84, 4)),
                # tf.keras.layers.Reshape((50*50, 84, 4)),

                # tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 120, 4)),
                tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 87, 4)),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv1'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                # tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 14)),
                tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 10)),

                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.MaxPooling1D(pool_size=16),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),
                top_k_pixel_pooling_class(k=cfg.top_k , pool_size=0, is_sorted=False),
                # tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k}),
                # top_k_pixel_pooling_class(k=10, is_sorted=False, input_dim=cfg.seq_in_samples // 32, pool_size=100),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="21_3")

        ##### 4 ##### stride encoder - top k - exp dens
        if (cfg.sub_model_type == 4):
            z = np.array([max(0.5 ** (i // 4), 1e-10) for i in range(4, 4 * cfg.top_k//2 + 4)], dtype=np.float64)
            z = np.concatenate((z, -1 * np.flipud(z))).reshape((4 * cfg.top_k, 1))

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 87, 4)),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv1'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 10)),

                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv1'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.MaxPooling1D(pool_size=32),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv2'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv3'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv4'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                # try_class(k=cfg.top_k),
                tf.keras.layers.Lambda(top_k_pixel_pooling_max, arguments={'k': cfg.top_k}),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=cfg.dropout),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1,dtype='float64',trainable=False, activation='linear', weights=[np.array([(1 - 2 * (i//(cfg.top_k*2))) * max(0.5**(i//4),1e-10) for i in range(4,4 * cfg.top_k+4)],dtype=np.float64).reshape((4 * cfg.top_k,1)), np.zeros(shape=(1),dtype=np.float64)],name='exp'),
                # # tf.keras.layers.Dense(1,dtype='float64',trainable=False, activation='linear', weights=[z, np.zeros(shape=(1),dtype=np.float64)],name='exp'),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="21_4")

            a = model.get_layer(name='exp').get_weights()
            print(a)

        #########   5   ######### stride encoder - concatinate point classifier with top_k classifier
        if (cfg.sub_model_type == 5):
            # encoder
            X_input = tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 87, 4))
            e = tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv1')(X_input)
            e = tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2')(e)
            e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3')(e)
            e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4')(e)
            e = tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU,
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5')(e)
            e = tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                   kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6')(e)
            e = tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 10))(e)
            # classifier
            c = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv1')(e)
            # c = tf.keras.layers.BatchNormalization(renorm=True, name='BN_1')(c)
            c = tf.keras.layers.MaxPooling1D(pool_size=32)(c)
            c = tf.keras.layers.Dropout(rate=cfg.dropout)(c)
            c = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv2')(c)
            c = tf.keras.layers.BatchNormalization(renorm=True, name='BN_2')(c)
            c = tf.keras.layers.Dropout(rate=cfg.dropout)(c)
            c = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv3')(c)
            c = tf.keras.layers.BatchNormalization(renorm=True, name='BN_3')(c)
            c = tf.keras.layers.Dropout(rate=cfg.dropout)(c)
            c = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv4')(c)
            c = tf.keras.layers.BatchNormalization(renorm=True, name='BN_4')(c)
            c = tf.keras.layers.Dropout(rate=cfg.dropout)(c)
            c = tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv5')(c)
            c = tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k})(c)
            c = tf.keras.layers.BatchNormalization(renorm=True, name='BN_5')(c)
            c = tf.keras.layers.Dropout(rate=cfg.dropout)(c)
            c = tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='C_conv6')(c)
            c = tf.keras.layers.BatchNormalization(renorm=True, name='BN_6')(c)
            c = tf.keras.layers.Dropout(rate=cfg.dropout + 0.1)(c)
            c = tf.keras.layers.Flatten(name='C_flatten')(c)

            # point
            p = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='P_conv1')(e)
            p = tf.keras.layers.BatchNormalization(renorm=True, name='BN_P_1')(p)
            p = tf.keras.layers.Dropout(rate=cfg.dropout)(p)
            p = tf.keras.layers.Conv1D(128, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                   bias_regularizer=l2(0.00001), name='P_conv2')(p)
            p = tf.keras.layers.BatchNormalization(renorm=True, name='BN_P_2')(p)
            p = tf.keras.layers.Dropout(rate=cfg.dropout)(p)
            p = tf.keras.layers.AveragePooling1D(pool_size=cfg.seq_in_samples * cfg.extra_sample)(p)
            # p = tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
            #                       bias_regularizer=l2(0.00001), name='P_dense1')(p)
            # p = tf.keras.layers.BatchNormalization(renorm=True, name='BN_P_3')(p)
            # p = tf.keras.layers.Dropout(rate=cfg.dropout)(p)
            p = tf.keras.layers.Flatten(name='P_flatten')(p)

            # concatinate C&P
            C_P = tf.keras.layers.concatenate([c, p],axis=1)

            # dense
            d = tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001))(C_P)
            d = tf.keras.layers.Dropout(rate=cfg.dropout + 0.1)(d)
            tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001))(d)
            # d = tf.keras.layers.Dropout(rate=cfg.dropout)(d)
            d = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))(d)

            model = Model(inputs=X_input, outputs=d, name='21_5')

        #########   6   ######### stride encoder - concatinate multiple top_k classifier with difrent maxpooling kernel
        if (cfg.sub_model_type == 6):
            # encoder
            X_input = tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 87, 4))
            e = tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv1')(X_input)
            e = tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2')(e)
            e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3')(e)
            e = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4')(e)
            e = tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5')(e)
            e = tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6')(e)
            e = tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 10))(e)
            # classifier

            # c1 4
            c1 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='1C_conv1')(e)
            c1 = tf.keras.layers.MaxPooling1D(pool_size=4)(c1)
            c1 = tf.keras.layers.Dropout(rate=cfg.dropout)(c1)
            c1 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='1C_conv2')(c1)
            c1 = tf.keras.layers.BatchNormalization(renorm=True, name='1BN_2')(c1)
            c1 = tf.keras.layers.Dropout(rate=cfg.dropout)(c1)
            c1 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='1C_conv3')(c1)
            c1 = tf.keras.layers.BatchNormalization(renorm=True, name='1BN_3')(c1)
            c1 = tf.keras.layers.Dropout(rate=cfg.dropout)(c1)
            c1 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='1C_conv4')(c1)
            c1 = tf.keras.layers.BatchNormalization(renorm=True, name='1BN_4')(c1)
            c1 = tf.keras.layers.Dropout(rate=cfg.dropout)(c1)
            c1 = tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='1C_conv5')(c1)
            c1 = tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k})(c1)
            c1 = tf.keras.layers.BatchNormalization(renorm=True, name='1BN_5')(c1)
            c1 = tf.keras.layers.Dropout(rate=cfg.dropout)(c1)
            c1 = tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='1C_conv6')(c1)
            c1 = tf.keras.layers.BatchNormalization(renorm=True, name='1BN_6')(c1)
            c1 = tf.keras.layers.Dropout(rate=cfg.dropout + 0.1)(c1)
            c1 = tf.keras.layers.Flatten(name='1C_flatten')(c1)

            # c2 8
            c2 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='2C_conv1')(e)
            c2 = tf.keras.layers.MaxPooling1D(pool_size=8)(c2)
            c2 = tf.keras.layers.Dropout(rate=cfg.dropout)(c2)
            c2 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='2C_conv2')(c2)
            c2 = tf.keras.layers.BatchNormalization(renorm=True, name='2BN_2')(c2)
            c2 = tf.keras.layers.Dropout(rate=cfg.dropout)(c2)
            c2 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='2C_conv3')(c2)
            c2 = tf.keras.layers.BatchNormalization(renorm=True, name='2BN_3')(c2)
            c2 = tf.keras.layers.Dropout(rate=cfg.dropout)(c2)
            c2 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='2C_conv4')(c2)
            c2 = tf.keras.layers.BatchNormalization(renorm=True, name='2BN_4')(c2)
            c2 = tf.keras.layers.Dropout(rate=cfg.dropout)(c2)
            c2 = tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='2C_conv5')(c2)
            c2 = tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k})(c2)
            c2 = tf.keras.layers.BatchNormalization(renorm=True, name='2BN_5')(c2)
            c2 = tf.keras.layers.Dropout(rate=cfg.dropout)(c2)
            c2 = tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='2C_conv6')(c2)
            c2 = tf.keras.layers.BatchNormalization(renorm=True, name='2BN_6')(c2)
            c2 = tf.keras.layers.Dropout(rate=cfg.dropout + 0.1)(c2)
            c2 = tf.keras.layers.Flatten(name='2C_flatten')(c2)

            # c3 16
            c3 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='3C_conv1')(e)
            c3 = tf.keras.layers.MaxPooling1D(pool_size=16)(c3)
            c3 = tf.keras.layers.Dropout(rate=cfg.dropout)(c3)
            c3 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='3C_conv2')(c3)
            c3 = tf.keras.layers.BatchNormalization(renorm=True, name='3BN_2')(c3)
            c3 = tf.keras.layers.Dropout(rate=cfg.dropout)(c3)
            c3 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='3C_conv3')(c3)
            c3 = tf.keras.layers.BatchNormalization(renorm=True, name='3BN_3')(c3)
            c3 = tf.keras.layers.Dropout(rate=cfg.dropout)(c3)
            c3 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='3C_conv4')(c3)
            c3 = tf.keras.layers.BatchNormalization(renorm=True, name='3BN_4')(c3)
            c3 = tf.keras.layers.Dropout(rate=cfg.dropout)(c3)
            c3 = tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='3C_conv5')(c3)
            c3 = tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k})(c3)
            c3 = tf.keras.layers.BatchNormalization(renorm=True, name='3BN_5')(c3)
            c3 = tf.keras.layers.Dropout(rate=cfg.dropout)(c3)
            c3 = tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='3C_conv6')(c3)
            c3 = tf.keras.layers.BatchNormalization(renorm=True, name='3BN_6')(c3)
            c3 = tf.keras.layers.Dropout(rate=cfg.dropout + 0.1)(c3)
            c3 = tf.keras.layers.Flatten(name='3C_flatten')(c3)

            # c4 32
            c4 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='4C_conv1')(e)
            c4 = tf.keras.layers.MaxPooling1D(pool_size=32)(c4)
            c4 = tf.keras.layers.Dropout(rate=cfg.dropout)(c4)
            c4 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='4C_conv2')(c4)
            c4 = tf.keras.layers.BatchNormalization(renorm=True, name='4BN_2')(c4)
            c4 = tf.keras.layers.Dropout(rate=cfg.dropout)(c4)
            c4 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='4C_conv3')(c4)
            c4 = tf.keras.layers.BatchNormalization(renorm=True, name='4BN_3')(c4)
            c4 = tf.keras.layers.Dropout(rate=cfg.dropout)(c4)
            c4 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='4C_conv4')(c4)
            c4 = tf.keras.layers.BatchNormalization(renorm=True, name='4BN_4')(c4)
            c4 = tf.keras.layers.Dropout(rate=cfg.dropout)(c4)
            c4 = tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='4C_conv5')(c4)
            c4 = tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k})(c4)
            c4 = tf.keras.layers.BatchNormalization(renorm=True, name='4BN_5')(c4)
            c4 = tf.keras.layers.Dropout(rate=cfg.dropout)(c4)
            c4 = tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='4C_conv6')(c4)
            c4 = tf.keras.layers.BatchNormalization(renorm=True, name='4BN_6')(c4)
            c4 = tf.keras.layers.Dropout(rate=cfg.dropout + 0.1)(c4)
            c4 = tf.keras.layers.Flatten(name='4C_flatten')(c4)

            # c5 64
            c5 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='5C_conv1')(e)
            c5 = tf.keras.layers.MaxPooling1D(pool_size=64)(c5)
            c5 = tf.keras.layers.Dropout(rate=cfg.dropout)(c5)
            c5 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='5C_conv2')(c5)
            c5 = tf.keras.layers.BatchNormalization(renorm=True, name='5BN_2')(c5)
            c5 = tf.keras.layers.Dropout(rate=cfg.dropout)(c5)
            c5 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='5C_conv3')(c5)
            c5 = tf.keras.layers.BatchNormalization(renorm=True, name='5BN_3')(c5)
            c5 = tf.keras.layers.Dropout(rate=cfg.dropout)(c5)
            c5 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='5C_conv4')(c5)
            c5 = tf.keras.layers.BatchNormalization(renorm=True, name='5BN_4')(c5)
            c5 = tf.keras.layers.Dropout(rate=cfg.dropout)(c5)
            c5 = tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='5C_conv5')(c5)
            c5 = tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k})(c5)
            c5 = tf.keras.layers.BatchNormalization(renorm=True, name='5BN_5')(c5)
            c5 = tf.keras.layers.Dropout(rate=cfg.dropout)(c5)
            c5 = tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='5C_conv6')(c5)
            c5 = tf.keras.layers.BatchNormalization(renorm=True, name='5BN_6')(c5)
            c5 = tf.keras.layers.Dropout(rate=cfg.dropout + 0.1)(c5)
            c5 = tf.keras.layers.Flatten(name='5C_flatten')(c5)

            # concatinate c1 c2 c3 c4 c5
            C_P = tf.keras.layers.concatenate([c1, c2, c3, c4, c5], axis=1)

            # dense
            d = tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001))(C_P)
            d = tf.keras.layers.Dropout(rate=cfg.dropout + 0.1)(d)
            tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                  bias_regularizer=l2(0.00001))(d)
            # d = tf.keras.layers.Dropout(rate=cfg.dropout)(d)
            d = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001))(d)

            model = Model(inputs=X_input, outputs=d, name='21_6')

        ##### 7 ##### stride encoder - top k - regular dens
        if (cfg.sub_model_type == 7):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 87, 4)),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv1'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 10)),

                # tf.keras.layers.Conv1D(10, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                #                        bias_regularizer=l2(0.00001), name='C_conv5'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                top_k_pixel_pooling_class(k=4, is_sorted=False, input_dim=cfg.seq_in_samples, pool_size=32),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.AveragePooling1D(pool_size=4),
                # tf.keras.layers.Conv1D(128, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                #                        bias_regularizer=l2(0.00001), name='C_conv6'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                # tf.keras.layers.Conv1D(256, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                #                        bias_regularizer=l2(0.00001), name='C_conv7'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                # tf.keras.layers.Conv1D(512, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                #                        bias_regularizer=l2(0.00001), name='C_conv8'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                # tf.keras.layers.MaxPooling1D(pool_size=cfg.seq_in_samples // 32 * 2),

                # tf.keras.layers.Conv1D(512, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                #                        bias_regularizer=l2(0.00001), name='C_conv9'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                # tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                #                        bias_regularizer=l2(0.00001), name='C_conv10'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                # tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(256, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_8'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_9'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="21_7")

        ##### 8 ##### stride encoder - top k - regular dens
        if (cfg.sub_model_type == 8):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 87, 4)),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv1'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 81)),

                top_k_pixel_pooling_class(k=1, is_sorted=False, input_dim=cfg.seq_in_samples, pool_size=8),
                # tf.keras.layers.AveragePooling1D(pool_size=4),
                tf.keras.layers.Lambda(matmul),
                tf.keras.layers.Reshape((cfg.seq_in_samples//8*1//1, cfg.seq_in_samples//8*1//1, 1)),

                tf.keras.layers.Conv2D(16, (3,3), strides=(2,2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv1'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(16, (3,3), strides=(2,2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv2'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(16, (3,3), strides=(2,2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv3'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(16, (3,3), strides=(2,2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv4'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(16, (3,3), strides=(2,2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_7'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_8'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(4, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_9'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv2D(1, (3, 3), strides=(2, 2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_10'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="21_8")

        #### 9 ##### top k pooling in encoder
        if (cfg.sub_model_type == 9):
            model = tf.keras.Sequential([
                # tf.keras.layers.Input(shape=(50, 50, 84, 4)),
                # tf.keras.layers.Reshape((50*50, 84, 4)),

                tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 87, 4)),
                tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, name='E_conv1'),
                tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2'),
                tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU,
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                top_k_pixel_pooling_class(k=1, pool_size=0,  is_sorted=False),

                tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 64)),

                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.MaxPooling1D(pool_size=32),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),
                top_k_pixel_pooling_class(k=100, pool_size=0,  is_sorted=False),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001))
            ], name="21_9")

        #### 10 ##### embedding layer
        if (cfg.sub_model_type == 10):
            model = tf.keras.Sequential([

                tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 19)),
                # tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 14)),
                tf.keras.layers.Embedding(input_dim=4**9 + 1, output_dim=64, input_length=19, embeddings_regularizer=l2(0.000001), name='embed1'),
                # tf.keras.layers.Embedding(input_dim=4**9 + 1, output_dim=64, input_length=14, embeddings_regularizer=l2(0.000001), name='embed1'),
                # tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU,
                #                        name='E_conv1'),
                # tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU,
                #                        kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                #                        name='E_conv2'),
                top_k_pixel_pooling_class(k=4, pool_size=0, is_sorted=False),
                tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 4*64)),

                # classifier

                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.MaxPooling1D(pool_size=32),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),

                top_k_pixel_pooling_class(k=cfg.top_k, pool_size=0, is_sorted=False),

                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001),activity_regularizer=l2(0.000001))
            ], name="21_10")

        return model

    if (cfg.model_type == 22):  # encoder-decoder/ embedding

        #### 1 ##### simple encoder decoder, encoder like 21_3
        if (cfg.sub_model_type == 1):
            model = tf.keras.Sequential([
                ### encoder ###
                tf.keras.layers.Input(shape=(cfg.seq_in_samples //4, 87, 4)),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv1'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                # tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 10)),

                ### decoder ###
                tf.keras.layers.Conv2DTranspose(32, (1, 3), strides=(1, 1), activation='relu', name='D_Tconv1'),
                tf.keras.layers.BatchNormalization(name='D_bn1'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', name='D_conv1'),

                tf.keras.layers.Conv2DTranspose(32, (1, 3), strides=(1, 2), activation='relu', name='D_Tconv2'),
                tf.keras.layers.BatchNormalization(name='D_bn2'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, name='D_conv2'),

                tf.keras.layers.Conv2DTranspose(32, (1, 3), strides=(1, 2), activation='relu', name='D_Tconv3'),
                tf.keras.layers.BatchNormalization(name='D_bn3'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, name='D_conv3'),

                tf.keras.layers.Conv2DTranspose(32, (1, 3), strides=(1, 2), activation='relu', name='D_Tconv4'),
                tf.keras.layers.BatchNormalization(name='D_bn4'),
                tf.keras.layers.Conv2D(16, (1, 4), strides=(1, 1), activation=cfg.LeakyReLU, name='D_conv4'),

                tf.keras.layers.Conv2DTranspose(32, (1, 3), strides=(1, 1), activation='relu', name='D_Tconv5'),
                tf.keras.layers.BatchNormalization(name='D_bn5'),
                tf.keras.layers.Conv2D(4, (1, 4), strides=(1, 1), activation=cfg.LeakyReLU, name='D_conv5'),

            ], name="22_1")

        #### 2 ##### simple encoder decoder, encoder like 21_3, decoder dense
        if (cfg.sub_model_type == 2):
            model = tf.keras.Sequential([
                ### encoder ###
                tf.keras.layers.Input(shape=(cfg.seq_in_samples // 4, 87, 4)),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv1'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv2'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv3'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv4'),
                tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv5'),
                tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv6'),
                tf.keras.layers.Reshape((cfg.seq_in_samples // 4, 10)),

                ### decoder ###
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001),name='D_dense1'),
                tf.keras.layers.BatchNormalization(name='D_bn1'),

                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='D_dense2'),
                tf.keras.layers.BatchNormalization(name='D_bn2'),

                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='D_dense3'),
                tf.keras.layers.BatchNormalization(name='D_bn3'),

                tf.keras.layers.Dense(256, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='D_dense4'),
                tf.keras.layers.BatchNormalization(name='D_bn4'),

                tf.keras.layers.Dense(348, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='D_dense5'),

                tf.keras.layers.Reshape((cfg.seq_in_samples // 4, 87,4)),

            ], name="22_2")

        #### 3 ##### simple encoder decoder, encoder dense, decoder dense
        if (cfg.sub_model_type == 3):
            model = tf.keras.Sequential([
                ### encoder ###
                tf.keras.layers.Input(shape=(cfg.seq_in_samples // 4, 87, 4)),
                tf.keras.layers.Reshape((cfg.seq_in_samples // 4, 348)),
                tf.keras.layers.Dense(256, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='E_dense1'),
                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='E_dense2'),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='E_dense3'),
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='E_dense4'),
                tf.keras.layers.Dense(10, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='E_dense5'),
                ### decoder ###
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='D_dense1'),
                tf.keras.layers.BatchNormalization(name='D_bn1'),

                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='D_dense2'),
                tf.keras.layers.BatchNormalization(name='D_bn2'),

                tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='D_dense3'),
                tf.keras.layers.BatchNormalization(name='D_bn3'),

                tf.keras.layers.Dense(256, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='D_dense4'),
                tf.keras.layers.BatchNormalization(name='D_bn4'),

                tf.keras.layers.Dense(348, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001), name='D_dense5'),

                tf.keras.layers.Reshape((cfg.seq_in_samples // 4, 87, 4)),

                tf.keras.layers.Dense(4, activation='softmax', name='D_dense6'),

            ], name="22_3")

        #### 4 ##### simple encoder decoder, encoder dense, decoder dense
        if (cfg.sub_model_type == 4):
            model = tf.keras.Sequential([
                ### encoder ###
                tf.keras.layers.Input(shape=(cfg.seq_in_samples // 4, 87*4)),
                tf.keras.layers.Embedding(input_dim=262200, output_dim=32, input_length=27),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv1'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv2'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv3'),
                # tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                #                        kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                #                        name='E_conv4'),
                # tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                #                        kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                #                        name='E_conv5'),
                # tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                #                        dilation_rate=(1, 1),
                #                        kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                #                        name='E_conv6'),
                # tf.keras.layers.Reshape((cfg.seq_in_samples // 4, 10)),
                # ### decoder ###
                # tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                #                       bias_regularizer=l2(0.00001), name='D_dense1'),
                # tf.keras.layers.BatchNormalization(name='D_bn1'),
                #
                # tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                #                       bias_regularizer=l2(0.00001), name='D_dense2'),
                # tf.keras.layers.BatchNormalization(name='D_bn2'),
                #
                # tf.keras.layers.Dense(128, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                #                       bias_regularizer=l2(0.00001), name='D_dense3'),
                # tf.keras.layers.BatchNormalization(name='D_bn3'),
                #
                # tf.keras.layers.Dense(256, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                #                       bias_regularizer=l2(0.00001), name='D_dense4'),
                # tf.keras.layers.BatchNormalization(name='D_bn4'),
                #
                # tf.keras.layers.Dense(348, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                #                       bias_regularizer=l2(0.00001), name='D_dense5'),
                #
                # tf.keras.layers.Reshape((cfg.seq_in_samples // 4, 87, 4)),
                #
                # tf.keras.layers.Dense(4, activation='softmax', name='D_dense6'),

            ], name="22_4")

        return model

    if (cfg.model_type == 23):  # global & local order

        #### 1 #####
        if (cfg.sub_model_type == 1):
            def custom_l2_regularizer(weights):
                return tf.reduce_sum(0.02 * tf.square(weights))

            class my_model(tf.keras.Model):
                def __init__(self):
                    super(my_model, self).__init__()
                    self.e0 = tf.keras.layers.Input(shape=(cfg.seq_in_samples * 4, 87, 4))
                    # metadata_indexes = tf.keras.layers.Input(shape=(cfg.seq_in_samples, 3))
                    # encoder
                    self.pool_size = 16
                    self.__name__="23_1"

                    # self.v1 = tf.compat.v1.get_variable('gate_mul', dtype=tf.float32,
                    #                                    initializer=np.zeros(shape=(cfg.seq_in_samples // self.pool_size, 1),
                    #                                                         dtype=np.float32) + 1.0, regularizer=None)
                    # self.v2 = tf.compat.v1.get_variable('gate_add', dtype=tf.float32,
                    #                                    initializer=np.zeros(shape=(cfg.seq_in_samples // self.pool_size, 1),
                    #                                                         dtype=np.float32) + 0.3, regularizer=None)

                    self.e1 = tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), input_shape=(cfg.seq_in_samples, 87, 4), activation=cfg.LeakyReLU, padding='same', name='E_conv1')
                    self.e2 = tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2')
                    self.e3 = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3')
                    self.e4 = tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4')
                    self.e5 = tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5')
                    self.e6 = tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6')
                    self.e7 = tf.keras.layers.Reshape((cfg.seq_in_samples, 10))
                    self.e8 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv1')
                    self.e9 = tf.keras.layers.MaxPooling1D(pool_size=self.pool_size)
                    self.e10 = tf.keras.layers.Dropout(rate=cfg.dropout)

                    self.c1 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv2')
                    self.c2 = tf.keras.layers.BatchNormalization(renorm=True)
                    self.c3 = tf.keras.layers.Dropout(rate=cfg.dropout)
                    self.c4 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv3')
                    self.c5 = tf.keras.layers.BatchNormalization(renorm=True)
                    self.c6 = tf.keras.layers.Dropout(rate=cfg.dropout)
                    self.c7 = tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv4')
                    self.c8 = tf.keras.layers.BatchNormalization(renorm=True)
                    self.c9 = tf.keras.layers.Dropout(rate=cfg.dropout)
                    self.c10 = tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv5')
                    self.top_k = top_k_pixel_pooling_class(k=cfg.top_k , pool_size=0, is_sorted=False)
                    # self.top_k = top_k_pixel_pooling_class_test_with_external_aditive_variables(k=cfg.top_k , pool_size=0, is_sorted=False)
                    self.c11 = tf.keras.layers.BatchNormalization(renorm=True)
                    self.c12 = tf.keras.layers.Dropout(rate=cfg.dropout)
                    self.c13 = tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='C_conv6')

                    self.d0 = tf.keras.layers.Flatten()
                    self.d1 = tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
                    self.d2 = tf.keras.layers.Dropout(rate=cfg.dropout + 0.1)
                    self.d3 = tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
                    self.d4 = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))

                def call(self, x):
                # def call(self, z):
                #     x, v = z[0], z[1]
                    # e = self.e0(x)
                    e = self.e1(x)
                    e = self.e2(e)
                    e = self.e3(e)
                    e = self.e4(e)
                    e = self.e5(e)
                    e = self.e6(e)
                    e = self.e7(e)
                    e = self.e8(e)
                    e = self.e9(e)
                    e = self.e10(e)

                    c = self.c1(e)
                    c = self.c2(c)
                    c = self.c3(c)
                    c = self.c4(c)
                    # c = tf.math.multiply(c, self.v1, name='gate_multiply')
                    # c = tf.math.add(c, self.v2, name='gate_multiply')
                    # v = tf.math.multiply(v, self.v2, name='test_multiply')
                    c = self.top_k(c)
                    # c = self.top_k(c,v)
                    c = self.c5(c)
                    c = self.c6(c)
                    c = self.c7(c)

                    d = self.d0(c)
                    d = self.d1(d)
                    d = self.d2(d)
                    d = self.d3(d)
                    d = self.d4(d)

                    return d

            model = my_model()

        #### 2 ##### adjustment in sample
        if (cfg.sub_model_type == 2):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(4441*4, 87, 4)),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       name='E_conv1'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv2'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv3'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv4'),
                tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv5'),
                tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001),
                                       name='E_conv6'),
                # tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 14)),
                tf.keras.layers.Reshape((4441*4, 10)),

                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv1'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.MaxPooling1D(pool_size=4),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv2'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 6, strides=3, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv3'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                # 2D
                tf.keras.layers.Conv1D(32, 6, strides=3, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv4'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(8, 6, strides=3, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(4, 6, strides=3, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(16, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Dense(16, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001))
            ], name="23_2")

        #### 3 ##### global order - dynamic - top k
        if (cfg.sub_model_type == 3):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(cfg.seq_in_samples * cfg.extra_sample, 87, 4)),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', name='E_conv1'),
                tf.keras.layers.Conv2D(32, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same', dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv2'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv3'),
                tf.keras.layers.Conv2D(16, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv4'),
                tf.keras.layers.Conv2D(8, (1, 3), strides=(1, 2), activation=cfg.LeakyReLU, dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv5'),
                tf.keras.layers.Conv2D(1, (1, 3), strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                                       dilation_rate=(1, 1),
                                       kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001), name='E_conv6'),
                # tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 14)),
                tf.keras.layers.Reshape((cfg.seq_in_samples * cfg.extra_sample, 10)),

                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv5'),
                # tf.keras.layers.BatchNormalization(renorm=True, name='BN_1'),
                tf.keras.layers.MaxPooling1D(pool_size=4),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv6'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_2'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv7'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_3'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(64, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv8'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_4'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(32, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv9'),
                top_k_pixel_pooling_class(k=cfg.top_k , pool_size=0, is_sorted=False),
                # tf.keras.layers.Lambda(top_k_pixel_pooling, arguments={'k': cfg.top_k}),
                # top_k_pixel_pooling_class(k=10, is_sorted=False, input_dim=cfg.seq_in_samples // 32, pool_size=100),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_5'),
                tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Conv1D(4, 1, strides=1, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                       bias_regularizer=l2(0.00001), name='C_conv10'),
                tf.keras.layers.BatchNormalization(renorm=True, name='BN_6'),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                tf.keras.layers.Dropout(rate=cfg.dropout + 0.1),
                tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                                      bias_regularizer=l2(0.00001)),
                # tf.keras.layers.Dropout(rate=cfg.dropout),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(0.00001), bias_regularizer=l2(0.00001))
            ], name="21_3")



        return model

def identity_block_3D(x, stage, block, filters=[64, 64, 64], conv_kernel=(1, 1, 2)):
    # defining name basis
    conv_name_base = 'res_stage_' + str(stage) + "_block_" + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    # save input value
    x_shortcut = x
    # first component of main path
    x = Conv3D(F1, conv_kernel, strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_a')(x)
    # x = BatchNormalization(axis=4, name=bn_name_base + '_a')(x)
    # second component of main path
    x = Conv3D(F2, conv_kernel, strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_b')(x)
    # x = BatchNormalization(axis=4, name=bn_name_base + '_b')(x)
    # third component of main path
    x = Conv3D(F3, conv_kernel, strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_c')(x)
    # x = BatchNormalization(axis=4, name=bn_name_base + '_c')(x)
    # add shortcut
    x = Add()([x, x_shortcut])
    x = cfg.LeakyReLU(x)
    return x


def conv_block_3D(x, stage, block, filters=[64, 64, 64], conv_kernel=(1, 1, 2)):
    # defining name basis
    conv_name_base = 'res_stage_' + str(stage) + "_block_" + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    # save input value
    x_shortcut = x
    # first component of main path
    x = Conv3D(F1, conv_kernel, strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_a')(x)
    # x = BatchNormalization(axis=4, name=bn_name_base + '_a')(x)
    # second component of main path
    x = Conv3D(F2, conv_kernel, strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_b')(x)
    # x = BatchNormalization(axis=4, name=bn_name_base + '_b')(x)
    # third component of main path
    x = Conv3D(F3, conv_kernel, strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_c')(x)
    # x = BatchNormalization(axis=4, name=bn_name_base + '_c')(x)
    # shortcut path
    x_shortcut = Conv3D(F3, conv_kernel, strides=(1, 1, 1), activation=cfg.LeakyReLU, padding='same',
                        name=conv_name_base + '_d')(x_shortcut)
    # x_shortcut = BatchNormalization(axis=4, name=bn_name_base + '_d')(x_shortcut)
    # add shortcut
    x = Add()([x, x_shortcut])
    x = cfg.LeakyReLU(x)
    return x


def identity_block_2D(x, stage, block, filters=[64, 64, 64], conv_kernel=(2, 2)):
    # defining name basis
    conv_name_base = 'res_stage_' + str(stage) + "_block_" + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    # save input value
    x_shortcut = x
    # first component of main path
    x = Conv2D(F1, conv_kernel, strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_a')(x)
    x = BatchNormalization(renorm=True, name=bn_name_base + '_a')(x)
    # second component of main path
    x = Conv2D(F2, conv_kernel, strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_b')(x)
    x = BatchNormalization(renorm=True, name=bn_name_base + '_b')(x)
    # third component of main path
    x = Conv2D(F3, conv_kernel, strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_c')(x)
    x = BatchNormalization(renorm=True, name=bn_name_base + '_c')(x)
    # add shortcut
    x = Add()([x, x_shortcut])
    x = cfg.LeakyReLU(x)
    return x


def conv_block_2D(x, stage, block, filters=[64, 64, 64], conv_kernel=(2, 2)):
    # defining name basis
    conv_name_base = 'res_stage_' + str(stage) + "_block_" + block + '_branch'
    bn_name_base = 'bn_' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    # save input value
    x_shortcut = x
    # first component of main path
    x = Conv2D(F1, conv_kernel, strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_a')(x)
    x = BatchNormalization(renorm=True, name=bn_name_base + '_a')(x)
    # second component of main path
    x = Conv2D(F2, conv_kernel, strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_b')(x)
    x = BatchNormalization(renorm=True, name=bn_name_base + '_b')(x)
    # third component of main path
    x = Conv2D(F3, conv_kernel, strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
               name=conv_name_base + '_c')(x)
    x = BatchNormalization(renorm=True, name=bn_name_base + '_c')(x)
    # shortcut path
    x_shortcut = Conv2D(F3, conv_kernel, strides=(1, 1), activation=cfg.LeakyReLU, padding='same',
                        name=conv_name_base + '_d')(x_shortcut)
    x_shortcut = BatchNormalization(renorm=True, name=bn_name_base + '_d')(x_shortcut)
    # add shortcut
    x = Add()([x, x_shortcut])
    x = cfg.LeakyReLU(x)
    return x


def sum(x):
    return tf.math.reduce_sum(x, axis=-2, keepdims=False)


def mat_mul(A, B):
    return tf.matmul(A, B)


def max_pixel_pooling(x, pool_size=2):
    weights = tf.math.reduce_sum(x, axis=-1, keepdims=True)
    # weights = weights / tf.math.reduce_max(weights)  # might help
    x = tf.math.add(x, weights)
    x = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(x)
    # x = x / tf.math.reduce_max(weights)  # might help
    return x


# def top_k_pixel_pooling(x):
#     print(x)
#     k = 2
#     weights = tf.math.reduce_sum(x, axis=-1, keepdims=False)
#     print(weights)
#     weights_shape = tf.shape(weights)   # need to change to constant
#     print(weights_shape)
#     top_values, top_indices = tf.math.top_k(tf.reshape(weights,(cfg.batch_size,-1)), k=k, sorted=True)
#     print(top_indices)
#     top_indices = tf.stack(
#         ((top_indices // weights_shape[2]), (top_indices % weights_shape[2])), -1)
#     print(top_indices)
#     tf.print(top_indices)
#     x = tf.gather_nd(x, top_indices, batch_dims=0)
#     print(x)
#     return x

# def top_k_pixel_pooling(x,k=1):
#     weights = tf.math.reduce_sum(x, axis=-1, keepdims=True)
#     weights_shape = tf.shape(weights)   # need to change to constant
#     top_values, top_indices = tf.math.top_k(tf.reshape(weights,(cfg.batch_size,-1,)), k=k, sorted=True)
#     top_indices = tf.stack(
#         ((top_indices // weights_shape[2]), (top_indices % weights_shape[2])), -1)
#     x = tf.gather_nd(x, top_indices, batch_dims=1)
#     return x

class try_class(tf.keras.Model):
    def __init__(self, k=1):
        super(try_class, self).__init__()
        self.k = k
        self.flatten = Flatten()
        self.BN1 = tf.keras.layers.BatchNormalization(renorm=True, name='BN1')
        self.BN2 = tf.keras.layers.BatchNormalization(renorm=True, name='BN2')
        self.BN3 = tf.keras.layers.BatchNormalization(renorm=True, name='BN3')
        self.d1 = tf.keras.layers.Dense(32, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                              bias_regularizer=l2(0.00001))
        self.d2 = tf.keras.layers.Dense(2, activation=cfg.LeakyReLU, kernel_regularizer=l2(0.00001),
                              bias_regularizer=l2(0.00001))

        # self.d2 = tf.keras.layers.Dense(1,dtype='float64',trainable=True, activation='linear', weights=[np.array([(max(0.5**(i//4),1e-10) for i in range(4,4 * cfg.top_k+4)],dtype=np.float64).reshape((4 * cfg.top_k,1)), np.zeros(shape=(1),dtype=np.float64)],name='exp'),

    def dinamic_exp_array(self,t,ab):
        a = tf.slice(ab, [0, 0], [cfg.batch_size, 1])
        b = tf.slice(ab, [0, 1], [cfg.batch_size, 1])
        exp_array = [a ** ((b * i)//4) for i in range(4, 4*self.k+4)]
        exp_array = tf.concat(exp_array,axis=1)
        m = tf.keras.layers.Multiply()([t, exp_array])
        s = tf.math.reduce_sum(m, axis=-1, keepdims=True)
        return s

    def top_k_pixel_pooling_max(self,x):
        weights = tf.math.reduce_max(x, axis=-1, keepdims=True)
        top_values, top_indices = tf.math.top_k(tf.reshape(weights, (cfg.batch_size, -1,)), k=self.k, sorted=True)
        top_indices = tf.keras.layers.Reshape([self.k, 1])(top_indices)
        x = tf.gather_nd(x, top_indices, batch_dims=1)
        return x

    def call(self, x):
        # x = self.BN1(x)
        t = self.top_k_pixel_pooling_max(x)
        t = self.flatten(t)

        ab = self.d1(t)
        # ab = self.BN2(ab)
        ab = self.d2(ab)
        # ab = self.BN3(ab)

        s = self.dinamic_exp_array(t=t, ab=ab)
        return s


def top_k_pixel_pooling_max(x, k=1):
    # x = tf.dtypes.cast(x, tf.float64)
    weights = tf.math.reduce_max(x, axis=-1, keepdims=True)
    top_values, top_indices = tf.math.top_k(tf.reshape(weights, (cfg.batch_size, -1,)), k=k, sorted=True)
    top_indices = tf.keras.layers.Reshape([k, 1])(top_indices)
    x = tf.gather_nd(x, top_indices, batch_dims=1)
    return x

def top_k_pixel_pooling(x, k=1):
    # x = tf.dtypes.cast(x, tf.float64)
    weights = tf.math.reduce_sum(x, axis=-1, keepdims=True)
    top_values, top_indices = tf.math.top_k(tf.reshape(weights, (cfg.batch_size, -1,)), k=k, sorted=False)
    top_indices = tf.keras.layers.Reshape([k, 1])(top_indices)
    x = tf.gather_nd(x, top_indices, batch_dims=1)
    return x

def matmul(x):
    x = tf.linalg.matmul(
        x, x, transpose_a=False, transpose_b=True, adjoint_a=False, adjoint_b=False,
        a_is_sparse=False, b_is_sparse=False, name=None
    )
    return x

class top_k_pixel_pooling_class_1D(tf.keras.Model):
    def __init__(self, k=1, pool_size=cfg.seq_in_samples, is_sorted=True):
        super(top_k_pixel_pooling_class_1D, self).__init__()
        self.k = k
        self.pool_size = pool_size
        self.is_sorted = is_sorted

    def build(self, input_shape):
        self.input_shape1 = input_shape
        self.indeces_offset = tf.convert_to_tensor(np.reshape(np.arange(self.input_shape1[-2]//self.pool_size, dtype=np.int32) * self.pool_size,(1,-1,1)))

    def call(self, x):
        weights = tf.math.reduce_sum(x, axis=-1, keepdims=True)
        weights = tf.keras.layers.Reshape((self.input_shape1[-2] // self.pool_size, self.pool_size))(weights)
        top_values, top_indices = tf.math.top_k(weights, k=self.k, sorted=self.is_sorted)
        top_indices = tf.keras.layers.Add()([self.indeces_offset, top_indices])
        top_indices = tf.keras.layers.Reshape((self.input_shape1[-2] // self.pool_size * self.k,1))(top_indices)
        x = tf.gather_nd(x, top_indices, batch_dims=1)
        return x

class top_k_pixel_pooling_class(tf.keras.Model): #1D/2D
    def __init__(self, k=1, pool_size=0, is_sorted=False):
        super(top_k_pixel_pooling_class, self).__init__()
        self.k = k
        self.pool_size = pool_size
        self.is_sorted = is_sorted

    def build(self, input_shape):
        self.input_shape1 = input_shape
        if(self.pool_size == 0):
            self.pool_size = self.input_shape1[-2]

        if(len(input_shape) == 3): # define 1D path
            self.indeces_offset = tf.convert_to_tensor(np.reshape(np.arange(self.input_shape1[-2] // self.pool_size, dtype=np.int32) * self.pool_size, (1, -1, 1)))
            self.reshape_weights = tf.keras.layers.Reshape((self.input_shape1[-2] // self.pool_size, self.pool_size))
            self.reshape_top_indices = tf.keras.layers.Reshape((self.input_shape1[-2] // self.pool_size * self.k,1))
            self.batch_dims=1

        if(len(input_shape) == 4): # define 2D path
            self.indeces_offset = tf.convert_to_tensor(np.reshape(np.arange(self.input_shape1[-2]//self.pool_size, dtype=np.int32) * self.pool_size,(1,1,-1,1)))
            self.reshape_weights = tf.keras.layers.Reshape((self.input_shape1[-3], self.input_shape1[-2] // self.pool_size, self.pool_size))
            self.reshape_top_indices = tf.keras.layers.Reshape((self.input_shape1[-3], self.input_shape1[-2] // self.pool_size * self.k,1))
            self.batch_dims=2

    def call(self, x):
        weights = tf.math.reduce_sum(x, axis=-1, keepdims=True) # sum of all last dim vectors
        weights = self.reshape_weights(weights) # reshape to pool_size to prepare to next step top k choose from each pool
        top_values, top_indices = tf.math.top_k(weights, k=self.k, sorted=self.is_sorted) # find top k in each pool
        top_indices = tf.keras.layers.Add()([self.indeces_offset, top_indices]) # add offset to indices by pool size jump steps
        top_indices = self.reshape_top_indices(top_indices) # reshape indices back to fit x with tf.gather_nd / consider using tf.boolean_mask instead
        x = tf.gather_nd(x, top_indices, batch_dims=self.batch_dims)
        return x

class top_k_pixel_pooling_class_test_with_external_aditive_variables(tf.keras.Model): #1D/2D
    def __init__(self, k=1, pool_size=0, is_sorted=False):
        super(top_k_pixel_pooling_class_test_with_external_aditive_variables, self).__init__()
        self.k = k
        self.pool_size = pool_size
        self.is_sorted = is_sorted

    def build(self, input_shape):
        self.input_shape1 = input_shape
        if(self.pool_size == 0):
            self.pool_size = self.input_shape1[-2]

        if(len(input_shape) == 3): # define 1D path
            self.indeces_offset = tf.convert_to_tensor(np.reshape(np.arange(self.input_shape1[-2] // self.pool_size, dtype=np.int32) * self.pool_size, (1, -1, 1)))
            self.reshape_weights = tf.keras.layers.Reshape((self.input_shape1[-2] // self.pool_size, self.pool_size))
            self.reshape_top_indices = tf.keras.layers.Reshape((self.input_shape1[-2] // self.pool_size * self.k,1))
            self.batch_dims=1

        if(len(input_shape) == 4): # define 2D path
            self.indeces_offset = tf.convert_to_tensor(np.reshape(np.arange(self.input_shape1[-2]//self.pool_size, dtype=np.int32) * self.pool_size,(1,1,-1,1)))
            self.reshape_weights = tf.keras.layers.Reshape((self.input_shape1[-3], self.input_shape1[-2] // self.pool_size, self.pool_size))
            self.reshape_top_indices = tf.keras.layers.Reshape((self.input_shape1[-3], self.input_shape1[-2] // self.pool_size * self.k,1))
            self.batch_dims=2

    def call(self, x, v):
        weights = tf.math.reduce_sum(x, axis=-1, keepdims=True)  # sum of all last dim vectors
        tf.print(weights.shape)
        tf.print(v.shape)
        weights = self.reshape_weights(v)  # reshape to pool_size to prepare to next step top k choose from each pool
        tf.print(weights.shape)
        top_values, top_indices = tf.math.top_k(weights, k=self.k, sorted=self.is_sorted)  # find top k in each pool
        top_indices = tf.keras.layers.Add()([self.indeces_offset, top_indices])  # add offset to indices by pool size jump steps
        top_indices = self.reshape_top_indices(top_indices)  # reshape indices back to fit x with tf.gather_nd / consider using tf.boolean_mask instead
        x = tf.gather_nd(x, top_indices, batch_dims=self.batch_dims)
        return x

    # def call(self, x, v):
    #     # weights = tf.math.reduce_sum(x, axis=-1, keepdims=True) # sum of all last dim vectors
    #     # tf.print(weights.shape)
    #     # weights = self.reshape_weights(weights) # reshape to pool_size to prepare to next step top k choose from each pool
    #     # tf.print(weights.shape)
    #     tf.print(v.shape)
    #     v = tf.reshape(v, (2000))
    #     tf.print(v.shape)
    #     # v = tf.expand_dims(v, axis=0)
    #     # tf.print(v.shape)
    #     # v = tf.concat([v,v], axis=0)
    #     # tf.print(v.shape)
    #     # v = self.reshape_weights(v) # reshape to pool_size to prepare to next step top k choose from each pool
    #     # tf.print(v.shape)
    #     top_values, top_indices = tf.math.top_k(v, k=self.k, sorted=self.is_sorted) # find top k in each pool
    #     tf.print(top_indices.shape)
    #     # top_indices = self.reshape_top_indices(top_indices) # reshape indices back to fit x with tf.gather_nd / consider using tf.boolean_mask instead
    #     top_indices = tf.reshape(top_indices,(100,1))
    #     # top_indices = tf.keras.layers.Reshape((100,1))(top_indices)
    #     tf.print(top_indices.shape)
    #     x = tf.gather_nd(x, top_indices, batch_dims=self.batch_dims)
    #     return x

class global_order_top_k_class(tf.keras.Model): #1D/2D
    def __init__(self, k=1, pool_size=0, is_sorted=False):
        super(global_order_top_k_class, self).__init__()
        self.k = k
        self.pool_size = pool_size
        self.is_sorted = is_sorted

    def build(self, input_shape):
        self.input_shape1 = input_shape
        if(self.pool_size == 0):
            self.pool_size = self.input_shape1[-2]

        if (len(input_shape) == 3):  # define 1D path
            self.reshape_weights = tf.keras.layers.Reshape((self.input_shape1[-2],))
            self.batch_dims = 1

        if(len(input_shape) == 4): # define 2D path
            self.batch_dims=2

    def call(self, x):
        weights = tf.math.reduce_sum(x, axis=-1, keepdims=True) # sum of all last dim vectors
        weights = self.reshape_weights(weights) # reshape to pool_size to prepare to next step top k choose from each pool
        top_values, top_indices = tf.math.top_k(weights, k=self.k, sorted=False) # find top k in whole sample
        x = tf.gather_nd(x, top_indices, batch_dims=self.batch_dims)
        return x

# @tf.function
def global_order(x,metadata_indexes, num_of_clasters):
    # z = np.zeros(shape=(1, x.shape[1]))
    z = tf.zeros(shape=(1, x.shape[-1]), dtype=tf.dtypes.float32)
    out = tf.zeros(shape=(1, x.shape[-1]), dtype=tf.dtypes.float32)
    pointer = 0
    for i in range(num_of_clasters):
        if (i == metadata_indexes[pointer,0]):
            # z[i] = x[metadata_indexes[pointer,1]:metadata_indexes[pointer,2]].max(axis=0)
            v = x[metadata_indexes[pointer,1]:metadata_indexes[pointer,2]]
            # v = np.expand_dims(v, axis=0)
            v = tf.expand_dims(v, axis=0)
            v = tf.keras.layers.GlobalMaxPool1D()(v)
            # v = v.max(axis=0)
            # v = tf.squeeze(v)
            # v2 = tf.keras.backend.eval(v)
            out = tf.keras.layers.concatenate([out, v], axis=0)
            pointer += 1
            if pointer == len(metadata_indexes):
                break
        else:
            out = tf.keras.layers.concatenate([out, z], axis=0)
    return out

# class top_k_pixel_pooling_class_2D(tf.keras.Model):
#     def __init__(self, k=1, pool_size=cfg.seq_in_samples, is_sorted=True):
#         super(top_k_pixel_pooling_class_2D, self).__init__()
#         self.k = k
#         self.pool_size = pool_size
#         self.is_sorted = is_sorted
#
#     def build(self, input_shape):
#         self.input_shape1 = input_shape
#         self.indeces_offset = tf.convert_to_tensor(np.reshape(np.arange(self.input_shape1[-2]//self.pool_size, dtype=np.int32) * self.pool_size,(1,1,-1,1)))
#
#
#     def call(self, x):
#         weights = tf.math.reduce_sum(x, axis=-1, keepdims=True)
#         weights = tf.keras.layers.Reshape((self.input_shape1[-3], self.input_shape1[-2] // self.pool_size, self.pool_size))(weights)
#         top_values, top_indices = tf.math.top_k(weights, k=self.k, sorted=self.is_sorted)
#         top_indices = tf.keras.layers.Add()([self.indeces_offset, top_indices])
#         top_indices = tf.keras.layers.Reshape((self.input_shape1[-3], self.input_shape1[-2] // self.pool_size * self.k,1))(top_indices)
#         x = tf.gather_nd(x, top_indices, batch_dims=2)
#         return x

# class top_k_pixel_pooling_class(tf.keras.Model):
#     def __init__(self, k=1, pool_size=cfg.seq_in_samples, batch_size=cfg.batch_size,seq_in_sample=cfg.seq_in_samples,
#                  input_dim=cfg.seq_in_samples, is_sorted=True):
#         super(top_k_pixel_pooling_class, self).__init__()
#         self.k = k
#         self.pool_size = pool_size
#         self.batch_size = batch_size
#         self.seq_in_sample = seq_in_sample
#         self.input_dim = input_dim
#         self.is_sorted = is_sorted
#         self.pool_indeces1 = np.expand_dims(np.repeat(np.arange(self.input_dim//self.pool_size,dtype=np.int32) * self.pool_size, self.k), axis=0)
#         self.pool_indeces2= np.repeat(np.arange(self.input_dim//self.pool_size,dtype=np.int32) * self.pool_size, self.k)
#         self.pool_indeces3= np.arange(self.input_dim//self.pool_size,dtype=np.int32) * self.pool_size
#         self.pool_indeces = np.repeat(np.expand_dims(np.repeat(np.arange(self.input_dim//self.pool_size,dtype=np.int32) * self.pool_size, self.k), axis=0), self.batch_size, axis=0)
#
#     def call(self, x):
#         weights = tf.math.reduce_sum(x, axis=-1, keepdims=True)
#         weights = tf.reshape(weights, (self.batch_size, self.input_dim // self.pool_size, self.pool_size))
#         top_values, top_indices = tf.math.top_k(weights, k=self.k, sorted=self.is_sorted)
#         top_indices = tf.keras.layers.Reshape([self.input_dim // self.pool_size* self.k])(top_indices)
#         top_indices = tf.math.add(self.pool_indeces, top_indices)
#         top_indices = tf.keras.layers.Reshape([self.input_dim // self.pool_size * self.k, 1])(top_indices)
#         x = tf.gather_nd(x, top_indices, batch_dims=1)
#         return x
#
#     def call_check3(self, x):
#         weights = tf.math.reduce_sum(x, axis=-1, keepdims=True)
#         weights = tf.reshape(weights, (self.batch_size, self.input_dim // self.pool_size, self.pool_size))
#         top_values, top_indices = tf.math.top_k(weights, k=self.k, sorted=self.is_sorted)
#         top_indices = tf.keras.layers.Reshape([self.input_dim // self.pool_size * self.k])(top_indices)
#         top_indices = tf.math.add(self.pool_indeces, top_indices)
#         top_indices = tf.keras.layers.Reshape([self.input_dim // self.pool_size * self.k, 1])(top_indices)
#         x = tf.gather_nd(x, top_indices, batch_dims=1)
#         return x
#
#     def call_check4(self, x):
#         weights = tf.math.reduce_sum(x, axis=-1, keepdims=True)
#         weights = tf.reshape(weights, (self.batch_size, self.seq_in_sample, self.input_dim // self.pool_size, self.pool_size))
#         top_values, top_indices = tf.math.top_k(weights, k=self.k, sorted=self.is_sorted)
#         top_indices = tf.keras.layers.Reshape([3,self.input_dim // self.pool_size * self.k])(top_indices)
#         top_indices = tf.math.add(self.pool_indeces, top_indices)
#         top_indices = tf.keras.layers.Reshape([self.input_dim // self.pool_size * self.k, 1])(top_indices)
#         x = tf.gather_nd(x, top_indices, batch_dims=1)
#         return x

def min_k_pixel_pooling(x, k=1):
    weights = tf.math.reduce_sum(x, axis=-1, keepdims=True) * (-1)
    top_values, top_indices = tf.math.top_k(tf.reshape(weights, (cfg.batch_size, -1,)), k=k, sorted=True)
    top_indices = tf.keras.layers.Reshape([k, 1])(top_indices)
    x = tf.gather_nd(x, top_indices, batch_dims=1)
    return x


def top_k_atention_pooling(x, w, k=1):
    top_values, top_indices = tf.math.top_k(tf.reshape(w, (cfg.batch_size, -1,)), k=k, sorted=True)
    top_indices = tf.keras.layers.Reshape([k, 1])(top_indices)
    x = tf.gather_nd(x, top_indices, batch_dims=1)
    return x


def top_k_atention_pooling_1(x, w, k=1):
    top_values, top_indices = tf.math.top_k(tf.reshape(w, (cfg.batch_size, -1,)), k=k, sorted=True)
    top_indices = tf.keras.layers.Reshape([k, 1])(top_indices)
    x = tf.gather_nd(x, top_indices, batch_dims=1)
    top_values = tf.keras.layers.Reshape([k, 1])(top_values)
    # soft_top_values = tf.keras.layers.Softmax(axis=-1)(top_values)
    # soft_top_values = tf.keras.layers.Reshape([k, 1])(soft_top_values)
    # x = tf.keras.layers.Multiply()([x, soft_top_values])
    x = tf.keras.layers.Multiply()([x, top_values])
    return x


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features):
        score = self.W2(self.W1(features))
        attention_weights = self.V(score)
        # attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        # context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

if __name__ == '__main__':
    # a = np.zeros((3, 12, 4))
    # a[0, 0, :] = 1
    # a[0, 1, :] = 2
    # a[0, 2, 0] = 10
    # a[1, 0, :] = 3
    # a[1, 2, :] = 4
    # a[1, 8, :] = 6
    # a[1, 10, :] = 5
    # a[2, 0, 0] = 0
    # a[2, 1, 0] = 1
    # a[2, 2, 0] = 2
    # a[2, 3, 0] = 3
    # a[2, 4, 0] = 4
    # a[2, 5, 0] = 5
    # a[2, 6, 0] = 6
    # a[2, 7, 0] = 7
    # a[2, 8, 0] = 8
    # a[2, 9, 0] = 9
    # a[2, 10, 0] = 10
    # a[2, 11, 0] = 11
    #
    # # class_top_k = top_k_pixel_pooling_class(k=2,pool_size=3, is_sorted=False)
    # class_top_k = global_order_top_k_class(k=2,pool_size=3, is_sorted=False)
    # a = np.reshape(np.arange(1*6*5), [1, 6, 5])
    # # class_top_k = top_k_pixel_pooling_class_2D(k=2,pool_size=3, is_sorted=True)
    # # a = np.reshape(np.arange(1*2*6*5), [1, 2, 6, 5])
    # # a = np.random.randint(100, size=(1, 2, 6, 1))
    # x = tf.convert_to_tensor(a, np.float32)
    # z = class_top_k(x)
    # z = tf.keras.backend.eval(z)
    # print(a)
    # print("z")
    # print(z)
    # # x = np.asarray(b, np.float32)
    # # x = tf.convert_to_tensor(x, np.float32)
    # # z = top_k_atention_pooling_1(x, w, k=2)
    # # z = class_top_k.call_check4(x)
    # # z = tf.keras.backend.eval(z)
    # # print(z)
    # # z = tf.keras.backend.eval(z)
    #
    # # global order tf function
    # x = np.random.randint(10, size=30).reshape((6,5)).astype(np.float32)
    # metadata_indexes = np.array([[1,0,3],[3,3,4],[8,4,6]])
    # print(x)
    # print(metadata_indexes)
    # num_of_clasters = 10
    # b = global_order(x, metadata_indexes, num_of_clasters)
    # b = tf.keras.backend.eval(b)
    # print(b)

    # x = np.random.randint(10, size=30).reshape((6,5)).astype(np.float32)
    # # metadata_indexes = np.array([[[0, 0, 0], [1, 0, 3], [2, 0, 0], [3, 3, 4], [4, 0, 0], [5, 0, 0], [6, 0, 0], [7, 0, 0], [8, 4, 6], [9, 0, 0]]])
    # metadata_indexes = np.array([[[0, 0], [0, 3], [0, 0], [3, 4], [0, 0], [0, 0], [0, 0], [0, 0], [4, 6], [0, 0]]])
    # global_class = global_order_class()
    # v = global_class(x,metadata_indexes)
    # v = tf.keras.backend.eval(v)

    # v = tf.compat.v1.get_variable(shape=(3, 1), initializer=tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.01), regularizer=tf.keras.regularizers.L2(
    # l2=0.01), dtype=tf.float32, name='gate')
    # e2 = tf.constant([[1.0,1.0],[2.0,2.0]])
    # print(tf.keras.backend.eval(v))
    # v = tf.sigmoid(v)
    # print(tf.keras.backend.eval(v))

    r = tf.ragged.constant([[[[0.3, 0.1, 0.4, 0.1], []], [[0.5, 0.9, 0.2], [0.6]], [[0.1],[]]]])
    c = tf.constant([[[0.3, 0.1, 0.4, 0.1], [0.0,0.0,0.0,0.0], [0.5, 0.9, 0.2,0.0], [0.6,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]]])
    model = tf.keras.Sequential([
        # tf.keras.layers.Conv1D(3, 1, strides=1),
                # tf.keras.layers.Input(shape=(1,5,)),
                tf.keras.layers.MaxPooling2D(pool_size=(1,2)),
            ])
    conv = tf.keras.layers.Conv1D(3, 3, strides=1)
    rt = r.to_tensor()
    z = conv(rt)
    tf.print(z)
    print('z')
    tf.print(model(rt))