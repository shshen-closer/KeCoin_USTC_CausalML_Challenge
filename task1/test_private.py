# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from datetime import datetime 
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from utils import checkmate as cmm
from utils import data_helpers as dh
import json

# Parameters
# ==================================================
#seq_len= int(sys.argv[1])
#batch_size = int(sys.argv[2])
logger = dh.logger_fn("tflog", "logs/test-{0}.log".format(time.asctime()).replace(':', '_'))
file_name = sys.argv[1]
path_name =  sys.argv[2]

MODEL = file_name
while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("The format of your input is illegal, it should be like(90175368), please re-input: ")
logger.info("The format of your input is legal, now loading to next step...")



MODEL_DIR =  path_name + '/' + MODEL + '/checkpoints/'
BEST_MODEL_DIR =  path_name + '/' + MODEL + '/bestcheckpoints1/'
SAVE_DIR = 'results/' + MODEL

# Data Parameters
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")
tf.compat.v1.flags.DEFINE_string("best_checkpoint_dir", BEST_MODEL_DIR, "Best checkpoint directory from training run")

# Model Hyperparameters
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.2, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", 500 , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("seq_len", 50, "Number of epochs to train for.")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
#logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
   #                             for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))
#


    

def test():

    # Load data
    logger.info("Loading data...")

    
    logger.info("test data processing...")
    test_students=[]
    for idx in range(5):

        data_index = str(idx)
        
        all_data =  pd.read_csv('../Task_1_dataset/Task_1_data_private_csv/dataset_' + data_index +'/train.csv', encoding = "ISO-8859-1", low_memory=False,header=None)
        
        array_data = np.array(all_data)
        
        for iii in range(0, 40000, 400):
            print(iii)
            temp = array_data[iii: iii+400]

            temp2 = []
            for ddd in temp:
                ddd = ddd[2:]
                fff = np.zeros(50)
                for ddds in range(50):
                    theone = int(ddd[ddds]*1000)
                    fff[ddds] = theone
                temp2.append(fff)
            temp2 = np.array(temp2, dtype= np.int16)

            c3 = temp[:,1]
            c3 = [int(jjj) for jjj in c3]

            for x in range(50):
                for y in range(50):
                    c1 = temp2[:,x]
                    c2 = temp2[:,y]

                    c11 = []
                    c22 = []
                    for jjj in c3:
                        if jjj == x:
                            c11.append(1)
                        else:
                            c11.append(0)
                        if jjj == y:
                            c22.append(1)
                        else:
                            c22.append(0)
                    c11 = np.array(c11, dtype= np.int16)
                    c22 = np.array(c22, dtype= np.int16)
                    test_students.append([c1, c2,c11, c22])
    leng = 400

    BEST_OR_LATEST = 'B'

    while not (BEST_OR_LATEST.isalpha() and BEST_OR_LATEST.upper() in ['B', 'L']):
        BEST_OR_LATEST = input("he format of your input is illegal, please re-input: ")
    if BEST_OR_LATEST == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cmm.get_best_checkpoint(FLAGS.best_checkpoint_dir, select_maximum_value=True)
    if BEST_OR_LATEST == 'L':
        logger.info("latest")
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            construct1 = graph.get_operation_by_name("construct1").outputs[0]
            construct2 = graph.get_operation_by_name("construct2").outputs[0]
            construct3 = graph.get_operation_by_name("construct3").outputs[0]
            construct4 = graph.get_operation_by_name("construct4").outputs[0]

            target_relation = graph.get_operation_by_name("target_relation").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]
            pred = graph.get_operation_by_name("pred").outputs[0]
            
            a=datetime.now()
            data_size = len(test_students)
            index = 0
            pred_labels = []
            
            while(index+FLAGS.batch_size <= data_size):
                construct1_b = np.zeros((FLAGS.batch_size, leng))
                construct2_b = np.zeros((FLAGS.batch_size, leng))
                construct3_b = np.zeros((FLAGS.batch_size, leng))
                construct4_b = np.zeros((FLAGS.batch_size, leng))
                target_relation_b = np.zeros((FLAGS.batch_size))

                for i in range(FLAGS.batch_size):
                    student = test_students[index+i]
                    construct1_b[i][:len(student[0])] = student[0]
                    construct2_b[i][:len(student[0])] = student[1]
                    construct3_b[i][:len(student[0])] = student[2]
                    construct4_b[i][:len(student[0])] = student[3]
                   # target_relation_b[i] = student[2]
                index += FLAGS.batch_size


                feed_dict = {
                    construct1: construct1_b,
                    construct2: construct2_b,
                    construct3: construct3_b,
                    construct4: construct4_b,
                    target_relation: target_relation_b,
                    dropout_keep_prob: 0.0,
                    is_training: False
                }

                pred_b = sess.run(pred, feed_dict)
                
                pred_labels.extend(pred_b.tolist())
            print(np.shape(pred_labels))
           
            pred_labels = np.reshape(pred_labels, [5,100*2500])
            pred_labels = np.reshape(pred_labels, [5,100,2500])
            pred_labels = np.mean(pred_labels, axis = 1)
           # pred_labels = np.greater_equal(pred_labels,0.455) 
           # pred_labels = pred_labels.astype(int)
            pred_labels = np.reshape(pred_labels, [5,50,50])
    np.save('results/adj_matrix' +file_name + '.npy', np.array(pred_labels))
    logger.info("Done.")


if __name__ == '__main__':
    test()
