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
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
plt.switch_backend('agg')

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

seed_value = cfg.seed
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device[5:6]
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# tf.compat.v1.disable_eager_execution()

out_path = "/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/results/" + cfg.save_dir + "/"
os.mkdir(out_path)
os.mkdir(out_path + '/logs')

# ### divide the repertoires to train, validation and test (for each folder in ordered_data folder) ###
# folders_path = '/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/ordered_data'
# folders_list = os.listdir(folders_path)
# for f in range(len(folders_list)):
#     files = os.listdir(folders_path + '/' + folders_list[f] + '/')
#     X_train, X_test = train_test_split(files, test_size=0.2, shuffle=True)
#     X_train, X_val = train_test_split(X_train, test_size=0.25)

def main():
    my_callbacks = [
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.9, patience=2, min_lr=0.000001),
        tf.keras.callbacks.EarlyStopping(patience=30,monitor="val_binary_accuracy"), # patience: Number of epochs with no improvement after which training will be stopped. I can put smaller patience so the early stopping will work
        tf.keras.callbacks.TensorBoard(log_dir=cfg.save_dir + '/logs')
    ]

    train_gen = Dataset.GeneratorOnline(mode="train", folds=cfg.X_train, batch_size=cfg.batch_size) # cfg.train_folds
    val_gen = Dataset.GeneratorOnline(mode="validation", folds=cfg.X_val, batch_size=cfg.batch_size) # cfg.val_folds

    z0 = train_gen.__getitem__(0)
    z0 = train_gen.__getitem__(0)

    # with tf.device('/cpu:0'):
    with tf.device(cfg.device):
        model = SequentialClassifier()

    model.summary()

    model.compile(cfg.optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])


    ### define weights of the unbalanced labels for the loss: ###
    path = cfg.data_path  #r"/home/home/dsi/moshe_sharabi/M.Sc/RepertoiresClassification/ordered_data_30000"  ## ordered_data_10000 / ordered_data_30000
    lable_list = []
    file_name = os.listdir(path)
    file_name_train = [s for s in file_name if any(xs in s for xs in cfg.X_train)]
    for file_name in file_name_train:
        if ("positive" in file_name):  ## positive or ill
            lable_list.append(1)
        else:
            lable_list.append(0)

    # for file_name in os.listdir(path):
    #     if (file_name[:2] in cfg.train_folds):  # generator holds file only from the requested folds
    #         if ("positive" in file_name):  ## positive or ill
    #             lable_list.append(1)
    #         else:
    #             lable_list.append(0)

    ### calc the weights to the classes
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(lable_list), y=lable_list)
    ### change weights from sklearn to tensorflow input
    class_weights = {l:c for l,c in zip(np.unique(lable_list), class_weights)}
    print("The weight for the classes are: ")
    print("\t- Class 0: "+str(class_weights[0]))
    print("\t- Class 1: "+str(class_weights[1]))

    with tf.device(cfg.device):
        history = model.fit(x=train_gen, epochs=cfg.max_epoch, verbose=1, callbacks=my_callbacks,
                            validation_data=val_gen, shuffle=True,
                            steps_per_epoch=cfg.steps_per_epoch, validation_steps=val_gen.__len__(), # steps_per_epoch: how many epochs I do in a train epoch. if I implement the earlystopping take a smaller value of steps_per_epoch
                            max_queue_size=cfg.max_queue_size, workers=cfg.workers,
                            use_multiprocessing=True, class_weight=class_weights)

    model.save_weights(cfg.weights_path_save, overwrite=True)
    print("weights_path:    " + cfg.weights_path_save)
    # plot(history)
    # for epoch in range(cfg.max_epoch):
    #     if (epoch % 400 == 0):
    # if (epoch % 1 == 0):
    #     model.load_weights(cfg.weights_path, by_name=True)
    #     print("weights loaded from:     ", cfg.weights_path)


    ##### plot the loss and AUC: #####

    history = history

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle("AUC and Loss - top K=100")
    # summarize history for acc
    ax1.plot(history.history['auc'])
    ax1.plot(history.history['val_auc'])
    ax1.set_ylabel('auc')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    # summarize history for loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    fig.savefig(out_path + cfg.weights_name + '_acc_loss.png')
    plt.close(fig)

if __name__ == '__main__':
    main()
