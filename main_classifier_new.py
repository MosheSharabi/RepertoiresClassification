import os
import logging
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
import numpy as np
from configs import cfg
from tempModel import SequentialClassifier
import Dataset

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

seed_value= cfg.seed
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device[5:6]
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# tf.compat.v1.disable_eager_execution()


def main():
    # loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    my_callbacks = [
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.9, patience=2, min_lr=0.000001),
        tf.keras.callbacks.EarlyStopping(patience=30,monitor="val_binary_accuracy"),
        tf.keras.callbacks.TensorBoard(log_dir=cfg.save_dir + '/logs')
    ]

    if(cfg.random_sample):
        train_gen = Dataset.GeneratorRandomFromUnitedFile(cfg.cleaned_path + r"/cubes/train/",batch_size=cfg.batch_size,
                                                          steps_per_epoch=5000)
        val_gen = Dataset.GeneratorRandomFromUnitedFile(cfg.cleaned_path + r"/cubes/validation/",batch_size=cfg.batch_size,
                                                        steps_per_epoch=500)
    elif(cfg.one_fold):
        train_gen = Dataset.GeneratorOnline(mode="train", folds=cfg.train_folds, batch_size=cfg.batch_size,
                                            augmentation=cfg.shaffel_augmentation,use_tags=cfg.use_tags,AE=cfg.AE)
        val_gen = Dataset.GeneratorOnline(mode="validation", folds=cfg.val_folds, batch_size=cfg.batch_size,
                                          use_tags=cfg.use_tags,AE=cfg.AE)
    else:
        train_gen = Dataset.GeneratorOnline(mode="train", batch_size=cfg.batch_size,
                                            augmentation=cfg.shaffel_augmentation,use_tags=cfg.use_tags)
        val_gen = Dataset.GeneratorOnline(mode="validation", batch_size=cfg.batch_size,use_tags=cfg.use_tags)

    z0 = train_gen.__getitem__(0)
    z0 = train_gen.__getitem__(0)

    # with tf.device('/cpu:0'):
    with tf.device(cfg.device):
        model = SequentialClassifier()
        # AdamOptimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)

    if (cfg.load_weights):
        w1 = model.get_weights()
        model.load_weights(cfg.weights_path, by_name=True)
        print("weights loaded from:     ", cfg.weights_path)
        w2 = model.get_weights()
        for a, b in zip(w1, w2):
            if np.all(a == b):
                print("wtf is happening")

    if (cfg.freeze_weights):
        print("freeze weights")
        model.get_layer(name='E_conv1').trainable = False
        model.get_layer(name='E_conv2').trainable = False
        model.get_layer(name='E_conv3').trainable = False
        model.get_layer(name='E_conv4').trainable = False
        model.get_layer(name='E_conv5').trainable = False
        model.get_layer(name='E_conv6').trainable = False

        model.get_layer(name='C_conv5').trainable = False
        model.get_layer(name='C_conv6').trainable = False
        model.get_layer(name='C_conv7').trainable = False
        model.get_layer(name='C_conv8').trainable = False
        model.get_layer(name='C_conv9').trainable = False
        model.get_layer(name='C_conv10').trainable = False

    model.summary()

    if (cfg.dataset == 'biomed'):

        model.compile(cfg.optimizer, loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()])
    elif ((cfg.dataset == 'celiac') | (cfg.dataset == 'cmv')):
        if (cfg.AE):
            model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.mse])
        else:
            model.compile(cfg.optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])



    with tf.device(cfg.device):
        # a = model.get_collection(tf.GraphKeys.LOSSES)
        # tf.compat.v1.add_to_collection('losses', tf.keras.regularizers.L1(l1=0.1)(model.v))
        # print(model.GraphKeys)
        # reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        # a =  sum(reg_losses)
        # v1 = model.v.numpy()
        # print(model.losses)
        # import math
        # @tf.function
        # def double_sinus_regularizer(x,a=0.01):
        #     return tf.reduce_sum(a * (0.5 *( tf.math.sin(1.5*np.pi+2*np.pi*x) + 1) -0.1*(tf.math.sin(0.5*np.pi+4*np.pi*x) - 1)))
        # def sinus_regularizer(x,a=0.01):
        #     return tf.reduce_sum(a * (0.5 + tf.math.sin(1.5*np.pi+2*np.pi*x) + 0.5))
        # def Gauss_regularizer(x,a=0.01,b=0.3,c=0):
        #     return tf.reduce_sum(a * (1/(b*(2*math.pi)**0.5) * math.e ** (-0.5*((x-c)/b)**2)))
        # def total_l2_regularizer(weights,a=0.00001):
        #     return a * tf.square(tf.reduce_sum(weights))
        #
        # regularizer = tf.keras.regularizers.l2(0.00001)
        # model.add_loss(lambda: regularizer(model.v1))
        # model.add_loss(lambda: regularizer(model.v2))
        # print(model.losses)
        history = model.fit(x=train_gen, epochs=cfg.max_epoch, verbose=1, callbacks=my_callbacks,
                            validation_data=val_gen, shuffle=True,
                            steps_per_epoch=cfg.steps_per_epoch, validation_steps=val_gen.__len__(),
                            max_queue_size=cfg.max_queue_size, workers=cfg.workers,
                            use_multiprocessing=True)
        # v = model.get_weights()[40]
        # for i in range(0,50):
        #     print(v[10*i+0][0],v[10*i+1][0],v[10*i+2][0],v[10*i+3][0],v[10*i+4][0],v[10*i+5][0],v[10*i+6][0],v[10*i+7][0],v[10*i+8][0],v[10*i+9][0])
        # print('min: ', model.get_weights()[40].min())
        # print('max: ', model.get_weights()[40].max())

        # print(model.v.get_weights())
        # v2 = model.v.numpy()
        # v2 = model.v
        # for i in range(30):
        #     tf.print(v2[i][0],' : ', v2[i][0])

    model.save_weights(cfg.weights_path_save,overwrite=True)
    print("weights_path:    " + cfg.weights_path_save)
    # plot(history)
    # for epoch in range(cfg.max_epoch):
    #     if (epoch % 400 == 0):
    # if (epoch % 1 == 0):
    #     model.load_weights(cfg.weights_path, by_name=True)
    #     print("weights loaded from:     ", cfg.weights_path)


if __name__ == '__main__':
    main()
