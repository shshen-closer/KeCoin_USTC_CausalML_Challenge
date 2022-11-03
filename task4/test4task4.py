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
from utils import checkmate as cm
from utils import data_helpers as dh

import json

# Parameters

logger = dh.logger_fn("tflog", "logs/test-{0}.log".format(time.asctime()).replace(':', '_'))
file_name = sys.argv[1]


MODEL = file_name
while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("The format of your input is illegal, it should be like(90175368), please re-input: ")
logger.info("The format of your input is legal, now loading to next step...")



MODEL_DIR =  'runs/' + MODEL + '/checkpoints/'
BEST_MODEL_DIR =  'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'results/' + MODEL

# Data Parameters
tf.compat.v1.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")
tf.compat.v1.flags.DEFINE_string("best_checkpoint_dir", BEST_MODEL_DIR, "Best checkpoint directory from training run")

# Model Hyperparameters
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.2, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_integer("hidden_size", 64, "The number of hidden nodes (Integer)")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", 32 , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("seq_len", 50, "Number of epochs to train for.")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100


    

def test():

    # Load data
    logger.info("Loading data...")

    
    logger.info("test data processing...")
    test_students1 = np.load("data/treat.npy", allow_pickle=True)
    test_students2 = np.load("data/control.npy", allow_pickle=True)
    test_students = np.concatenate([test_students1, test_students2, test_students1[:100]], axis = 0)
    jiedian = np.load("data/jiedian.npy", allow_pickle=True)
    jiedian_c = np.load("data/jiedian_c.npy", allow_pickle=True)

    allsum = np.sum(jiedian)
    print(allsum)
    allsum_c = np.sum(jiedian_c)
    print(allsum_c)

    max_num_steps = 200
    max_num_skills = 1075
    BEST_OR_LATEST = 'B'

    while not (BEST_OR_LATEST.isalpha() and BEST_OR_LATEST.upper() in ['B', 'L']):
        BEST_OR_LATEST = input("he format of your input is illegal, please re-input: ")
    if BEST_OR_LATEST == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(FLAGS.best_checkpoint_dir, select_maximum_value=True)
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
            input_problem = graph.get_operation_by_name("input_problem").outputs[0]
            input_kc = graph.get_operation_by_name("input_kc").outputs[0]

            x_answer = graph.get_operation_by_name("x_answer").outputs[0]
            target_id = graph.get_operation_by_name("target_id").outputs[0]
            target_kc = graph.get_operation_by_name("target_kc").outputs[0]

            target_index = graph.get_operation_by_name("target_index").outputs[0]
            target_correctness = graph.get_operation_by_name("target_correctness").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]
            pred = graph.get_operation_by_name("preds").outputs[0]
            
        
            a=datetime.now()
            data_size = len(test_students)
            index = 0
            actual_labels = []
            pred_labels = []
            
            leng = []
            while(index+FLAGS.batch_size <= data_size):
                input_problem_b = np.zeros((FLAGS.batch_size, max_num_steps))
                input_kc_b = np.ones((FLAGS.batch_size, max_num_steps, max_num_skills)) * 0.01
                x_answer_b = np.zeros((FLAGS.batch_size, max_num_steps))
                target_id_b = np.zeros((FLAGS.batch_size, max_num_steps))
                target_kc_b = np.ones((FLAGS.batch_size, max_num_steps, max_num_skills)) * 0.01
                target_correctness_b = []
                target_index_b = []
                pred_idx = []
                for i in range(FLAGS.batch_size):
                    student = test_students[index+i]
                    problem_ids = student[0]
                    correctness = student[1]
                    problem_kcs = student[2]
                    len_seq = student[3]
                    for j in range(len_seq-1):

                        input_problem_b[i,j] = problem_ids[j]
                        input_kc_b[i, j, int(problem_kcs[j])] = 1
                        x_answer_b[i,j] = correctness[j]
                        target_id_b[i,j] = problem_ids[j + 1]
                        target_kc_b[i, j, int(problem_kcs[j+1])] = 1
                        target_index_b.append(i*max_num_steps+j)
                        target_correctness_b.append(int(correctness[j+1]))
                        actual_labels.append(int(correctness[j+1]))
                    pred_idx.append(i*max_num_steps+j)

                index += FLAGS.batch_size


                feed_dict = {
                    input_problem: input_problem_b,
                    input_kc: input_kc_b,
                    x_answer: x_answer_b,
                    target_id: target_id_b,
                    target_kc: target_kc_b,
                    target_index: target_index_b,
                    target_correctness: target_correctness_b,
                    dropout_keep_prob: 0.0,
                    is_training: False
                }
                
                pred_b = sess.run(pred, feed_dict)
                for sss in pred_idx:
                    pred_labels.append(pred_b[sss])
                if len(pred_labels)% 10000 == 0:
                    print(len(pred_labels))
    pred_labels = pred_labels[:allsum + allsum_c]
    pred1 = pred_labels[:allsum]
    pred2 = pred_labels[allsum:]
    print(len(pred2))
    start = 0
    predicts1 = []
    for lll in jiedian:
        predicts1.append(np.mean(pred1[start:start + lll]))
        start += lll
    start = 0
    predicts2 = []
    for lll in jiedian_c:
        predicts2.append(np.mean(pred2[start:start + lll]))
        start += lll
    print(len(predicts1))
    print(len(predicts2))
    predicts = np.array(predicts1) - np.array(predicts2)
    print(predicts)
    np.save('cate_estimate.npy', np.array(predicts))   
            

    logger.info("Done.")


if __name__ == '__main__':
    test()
