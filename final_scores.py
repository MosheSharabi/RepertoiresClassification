import os
import logging
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
import numpy as np
from sklearn import metrics
from configs import cfg
from tempModel import SequentialClassifier
import Dataset
from sklearn.metrics import classification_report
from configs import cfg

# in addition to do 1: function that say what is the best amount of sub sample needed to decide if the whole rep is true
#  2: maybe to check what the minimum overall confidence the model need through all sub samples to decide if the whole rep is true

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

seed_value = cfg.seed
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
os.environ['PYTHONHASHSEED']=str(seed_value)
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device[5:6]
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# tf.compat.v1.disable_eager_execution()

# def dic_add_default(dic,element, default_value=1):
#     if element in dic:
#         dic[element] += default_value
#     else:
#         dic[element] = default_value
#     return dic

# def evaluate(model, gen):
#     scores = np.empty(3)
#     for i in range(len(gen)):
#         x, y, file_names = gen.getitem_with_fileName(i)
#         s = model.evaluate(x,y)
#         scores = np.add(scores, s)
#
#     print(scores / len(gen))

def get_predictions(model, gen):
    all_file_names = np.empty(0)
    all_y = np.empty(0)
    all_predictions = np.empty(0)

    for i in range(len(gen)):
    # for i in range(100):
        x, y, file_names = gen.getitem_with_fileName(i)
        predictions = model.predict_on_batch(x)  #predict_on_batch, on the other hand, assumes that the data you pass in is exactly one batch and thus feeds it to the network. It won't try to split it
        all_file_names = np.append(all_file_names,file_names)
        all_y = np.append(all_y,y)
        all_predictions = np.append(all_predictions,predictions)
    return all_file_names, all_y, all_predictions

def get_repertoires_predictions(all_file_names, all_y, all_predictions, threshold=0.50):
    all_predictions_rounded = np.where(all_predictions > threshold , 1, 0)
    unique_names = set([name[name.find("_") + 1:] for name in all_file_names])
    scores = {name: [0,0,0,0] for name in unique_names}
    for i in range(len(all_y)):
        name = all_file_names[i][all_file_names[i].find("_") + 1:]
        scores[name][0] = all_y[i]
        scores[name][1] += all_predictions_rounded[i]
        scores[name][2] += all_predictions[i]
        scores[name][3] += 1

    y = np.array([scores[name][0] for name in scores])
    pred1 = np.array([scores[name][1]/scores[name][3] for name in scores])
    pred2 = np.array([scores[name][2]/scores[name][3] for name in scores])

    find_optimal_threshold_scorses(y, pred1)
    find_optimal_threshold_scorses(y, pred2)

def find_optimal_threshold_scorses(y, pred): # convert the accuracy from the level of sub_clusters to the level of the repertoires
    scores = np.zeros((100,3))
    for i in range(100):
        temp_preds = np.where(pred > i/100, 1, 0)
        scores[i] = [metrics.accuracy_score(y, temp_preds), metrics.balanced_accuracy_score(y, temp_preds), metrics.roc_auc_score(y, pred)]

    print('threshold , acc , balanced_acc , auc')
    i = np.argmax(scores[:,0])
    print(i/100, scores[i,0],scores[i,1],scores[i,2])

def main():
    # train_gen = Dataset.GeneratorOnline(mode="train", folds=cfg.train_folds, batch_size=cfg.batch_size)
    test_gen = Dataset.GeneratorOnline(mode="test", folds=cfg.X_test, batch_size=cfg.batch_size) #val_folds , mode='validation' ,cfg.test_folds

    with tf.device(cfg.device):
        model = SequentialClassifier()

    if (cfg.load_weights):
        w1 = model.get_weights()
        model.load_weights(cfg.weights_path, by_name=True)
        print("weights loaded from:     ", cfg.weights_path)
        w2 = model.get_weights()
        for a, b in zip(w1, w2):
            if np.all(a == b):
                print("what is happening")

    model.summary()

    with tf.device(cfg.device):
        model.compile(cfg.optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])

        all_file_names, all_y, all_predictions = get_predictions(model, test_gen)

        # scores = [),
        #           ),
        #           metrics.roc_auc_score(all_y, all_predictions.round())]


        print(metrics.roc_auc_score(all_y, all_predictions.round()))
        type(metrics.roc_auc_score(all_y, all_predictions.round()))

        np.savetxt(cfg.weights_path + '_save_test_accuracy.txt', [metrics.accuracy_score(all_y, all_predictions.round())])
        np.savetxt(cfg.weights_path + '_save_test_balanced_accuracy.txt', [metrics.balanced_accuracy_score(all_y, all_predictions.round())])
        np.savetxt(cfg.weights_path + '_save_test_auc.txt', [metrics.roc_auc_score(all_y, all_predictions.round())])




        # find_optimal_threshold_scorses(all_y, all_predictions)
        # get_repertoires_predictions(all_file_names, all_y, all_predictions)
        # evaluate(model, val_gen)

        # make predictions on the testing images, finding the index of the
        # label with the corresponding largest predicted probability
        # predIdxs = model.predict(x=test_gen)
        # predIdxs = np.argmax(predIdxs, axis=1)
        # # show a nicely formatted classification report
        # print("[INFO] evaluating network...")
        # print(classification_report(testLabels.argmax(axis=1), predIdxs,
        #                             target_names=lb.classes_))


if __name__ == '__main__':
    main()



