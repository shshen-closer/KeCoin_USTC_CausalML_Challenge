# -*- coding:utf-8 -*-
__author__ = 'shshen'

import os
import sys
import time
import logging
import random
import tensorflow as tf
from datetime import datetime 
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import r2_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from model  import model1
from utils import checkmate as cm
from utils import data_helpers as dh
from sklearn.preprocessing import MinMaxScaler

def is_there_adjacency(adj_matrix):
    """
    If input is (n,n), this returns a 1D array of size n*(n-1)/2 indicating whether each edge is present or not (not
    considering orientation).
    """
    mask = np.tri(adj_matrix.shape[0], k=-1, dtype=bool)
    is_there_forward = adj_matrix[mask].astype(bool)
    is_there_backward = (adj_matrix.T)[mask].astype(bool)
    return is_there_backward | is_there_forward
def get_adjacency_type(adj_matrix):
    """
    If input is (n,n), this returns a 1D array of size n*(n-1)/2 indicating the type of each edge (that is, 0 if
    there is no edge, 1 if it is forward, -1 if it is backward and 2 if it is in both directions or undirected).
    """

    def aux(f, b):
        if f and b:
            return 2
        elif f and not b:
            return 1
        elif not f and b:
            return -1
        elif not f and not b:
            return 0

    mask = np.tri(adj_matrix.shape[0], k=-1, dtype=bool)
    is_there_forward = adj_matrix[mask].astype(bool)
    is_there_backward = (adj_matrix.T)[mask].astype(bool)
    out = np.array([aux(f, b) for (f, b) in zip(is_there_forward, is_there_backward)])
    return out


# 0.5 rmse 0.410292  auc 0.732877  r2 0.134572   acc0.753765 
# 0.8 max: rmse 0.410135  auc 0.733694  r2 0.135531   acc0.754367
# 0.2 max: rmse 0.410885  auc 0.731869  r2 0.132518   acc0.753098
# ==================================================

TRAIN_OR_RESTORE = 'T' #input("Train or Restore?(T/R): ")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input("The format of your input is illegal, please re-input: ")
logging.info("The format of your input is legal, now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()).replace(':', '_'))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()).replace(':', '_'))

number = str(sys.argv[1])
tf.compat.v1.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")
tf.compat.v1.flags.DEFINE_float("learning_rate", 0.0003, "Learning rate")
tf.compat.v1.flags.DEFINE_float("norm_ratio", 10, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.compat.v1.flags.DEFINE_float("keep_prob", 0.1, "Keep probability for dropout")
tf.compat.v1.flags.DEFINE_integer("hidden_size", 128, "The number of hidden nodes (Integer)")
tf.compat.v1.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.compat.v1.flags.DEFINE_integer("batch_size", 500 , "Batch size for training.")
tf.compat.v1.flags.DEFINE_integer("epochs", 15, "Number of epochs to train for.")


tf.compat.v1.flags.DEFINE_integer("decay_steps", 8, "how many steps before decay learning rate. (default: 500)")
tf.compat.v1.flags.DEFINE_float("decay_rate", 0.1, "Rate of decay for learning rate. (default: 0.95)")
tf.compat.v1.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 1000)")
tf.compat.v1.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 50)")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.compat.v1.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.compat.v1.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100


def train():
    """Training model."""

    # Load sentences, labels, and training parameters
    logger.info("Loading data...")

    logger.info("Training data processing...")
   # all_students = np.load("data/output.npy", allow_pickle=True)


    idxs = [0,1,2,3,4]
    truth = np.load('../Task_1_dataset/Task_1_data_local_dev_csv/adj_matrix.npy', allow_pickle=True)
    idxs.remove(int(number))
    leng = 400
    train_students=[]
    for idx in idxs:
       # print(idx)
        data_index = str(idx)
        all_data =  pd.read_csv('../Task_1_dataset/Task_1_data_local_dev_csv/dataset_' + data_index +'/train.csv', encoding = "ISO-8859-1", low_memory=False,header=None)
        truth = np.load('../Task_1_dataset/Task_1_data_local_dev_csv/adj_matrix.npy', allow_pickle=True)
        array_data = np.array(all_data)
        for iii in range(0, 40000, 400):
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
                            c22.append(0)
                        elif jjj == y:
                            c22.append(1)
                            c11.append(0)
                        else:
                            c22.append(0)
                            c11.append(0)
                    c11 = np.array(c11, dtype= np.int16)
                    c22 = np.array(c22, dtype= np.int16)
                    label = int(truth[int(data_index)][x][y])
                    train_students.append([c1, c2, label, c11, c22])

    valid_students=[]
    data_index = str(number)
    all_data =  pd.read_csv('../Task_1_dataset/Task_1_data_local_dev_csv/dataset_' + data_index +'/train.csv', encoding = "ISO-8859-1", low_memory=False,header=None)
    truth = np.load('../Task_1_dataset/Task_1_data_local_dev_csv/adj_matrix.npy', allow_pickle=True)
    array_data = np.array(all_data)
    for iii in range(0, 40000, 400):
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
                        c22.append(0)
                
                    elif jjj == y:
                        c22.append(1)
                        c11.append(0)
                    else:
                        c22.append(0)
                        c11.append(0)
                c11 = np.array(c11, dtype= np.int16)
                c22 = np.array(c22, dtype= np.int16)


                label = int(truth[int(data_index)][x][y])
                valid_students.append([c1, c2, label, c11, c22])
  #  train_students = np.concatenate([train_students, valid_students], axis = 0)
    train_students = np.array(train_students)
    valid_students = np.array(valid_students)
    np.random.shuffle(train_students)

   # valid_students = train_students[int(0.8*len(train_students)):]
   # train_students = train_students[:int(0.8*len(train_students))]
    print(np.shape(train_students))
    print(np.shape(valid_students))

    print((len(train_students)//FLAGS.batch_size + 1) * FLAGS.decay_steps)
    # Build a graph and lstm_3 object
    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            mmm = model1(
                batch_size = FLAGS.batch_size,
                hidden_size = FLAGS.hidden_size, 
                leng = leng,
                )
            

            # Define training procedure
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=mmm.global_step, decay_steps=(len(train_students)//FLAGS.batch_size +1) * FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
               # learning_rate = tf.train.piecewise_constant(FLAGS.epochs, boundaries=[7,10], values=[0.005, 0.0005, 0.0001])
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
               # grads, vars = zip(*optimizer.compute_gradients(mmm.loss))
                #grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                #train_op = optimizer.apply_gradients(zip(grads, vars), global_step=mmm.global_step, name="train_op")
                train_op = optimizer.minimize(mmm.loss, global_step=mmm.global_step, name="train_op")

            # Output directory for models and summaries
            if FLAGS.train_or_restore == 'R':
                MODEL = input("Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input("The format of your input is illegal, please re-input: ")
                logger.info("The format of your input is legal, now loading to next step...")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                logger.info("Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir1 = os.path.abspath(os.path.join(out_dir, "bestcheckpoints1"))
            best_checkpoint_dir2 = os.path.abspath(os.path.join(out_dir, "bestcheckpoints2"))

            # Summaries for loss
            loss_summary = tf.compat.v1.summary.scalar("loss", mmm.loss)

            # Train summaries
            train_summary_op = tf.compat.v1.summary.merge([loss_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.compat.v1.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.compat.v1.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            best_saver1 = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir1, num_to_keep=1, maximize=True)
            best_saver2 = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir2, num_to_keep=1, maximize=True)

            if FLAGS.train_or_restore == 'R':
                # Load mmm model
                logger.info("Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.compat.v1.global_variables_initializer())
                sess.run(tf.compat.v1.local_variables_initializer())



            current_step = sess.run(mmm.global_step)

            def train_step(construct1, construct2, construct3, construct4, target_relation):
                """A single training step"""
                
                feed_dict = {
                    mmm.construct1: construct1,
                    mmm.construct2: construct2,
                    mmm.construct3: construct3,
                    mmm.construct4: construct4,
                    mmm.target_relation: target_relation,
 
                    mmm.dropout_keep_prob: FLAGS.keep_prob,
                    mmm.is_training: True
                }
                _, step, summaries, pred, loss = sess.run(
                    [train_op, mmm.global_step, train_summary_op, mmm.pred, mmm.loss], feed_dict)

                
                logger.info("step {0}: loss {1:g} ".format(step,loss))
                train_summary_writer.add_summary(summaries, step)
                return pred, loss

            def validation_step(construct1, construct2, construct3, construct4, target_relation):
                """Evaluates model on a validation set"""

                feed_dict = {
                    mmm.construct1: construct1,
                    mmm.construct2: construct2,
                    mmm.construct3: construct3,
                    mmm.construct4: construct4,
                    mmm.target_relation: target_relation,
                    mmm.dropout_keep_prob: 0.0,
                    mmm.is_training: False
                }
                step, summaries, pred, loss = sess.run(
                    [mmm.global_step, validation_summary_op, mmm.pred,  mmm.loss], feed_dict)
                validation_summary_writer.add_summary(summaries, step)
                
                return pred, loss
            # Training loop. For each batch...
            
            run_time = []

         
            for iii in range(FLAGS.epochs):
                np.random.seed(iii*100)
                np.random.shuffle(train_students)
                a=datetime.now()
                data_size = len(train_students)
                index = 0
                actual_labels = []
                pred_labels = []
                losses = []
                while(index+FLAGS.batch_size <= data_size):
                    construct1 = np.zeros((FLAGS.batch_size, leng))
                    construct2 = np.zeros((FLAGS.batch_size, leng))
                    construct3 = np.zeros((FLAGS.batch_size, leng))
                    construct4 = np.zeros((FLAGS.batch_size, leng))
                    target_relation = np.zeros((FLAGS.batch_size))
                    
                    for i in range(FLAGS.batch_size):
                        student = train_students[index+i]
                        construct1[i][:len(student[0])] = student[0]
                        construct2[i][:len(student[0])] = student[1]
                        construct3[i] = student[3]
                        construct4[i] = student[4]
                        target_relation[i] = student[2]
                        actual_labels.append(student[2])
                   # print(current_construct)
                    index += FLAGS.batch_size
                    
                    pred, loss = train_step(construct1, construct2, construct3, construct4, target_relation)
                    losses.append(loss)
                    for p in pred:
                        pred_labels.append(p)
                    current_step = tf.compat.v1.train.global_step(sess, mmm.global_step)
                b=datetime.now()
                e_time = (b-a).total_seconds()
                run_time.append(e_time)


                logger.info("epochs {0}: loss {1:g}".format((iii +1), np.mean(losses)))
                
                

                if((iii+1) % FLAGS.evaluation_interval == 0):
                    logger.info("\nEvaluation:")
                    
                    data_size = len(valid_students)
                    index = 0
                    actual_labels = []
                    pred_labels = []
                    losses = []
                    while(index+FLAGS.batch_size <= data_size):
                        construct1 = np.zeros((FLAGS.batch_size, leng))
                        construct2 = np.zeros((FLAGS.batch_size, leng))
                        construct3 = np.zeros((FLAGS.batch_size, leng))
                        construct4 = np.zeros((FLAGS.batch_size, leng))
                        target_relation = np.zeros((FLAGS.batch_size))
                        for i in range(FLAGS.batch_size):
                            student = valid_students[index+i]
                            construct1[i][:len(student[0])] = student[0]
                            construct2[i][:len(student[0])] = student[1]
                            construct3[i] = student[3]
                            construct4[i] = student[4]
                            target_relation[i] = student[2]
                            actual_labels.append(student[2])
                        index += FLAGS.batch_size
                        pred,v_loss  = validation_step(construct1, construct2, construct3, construct4, target_relation)
                        losses.append(v_loss)
                        for p in pred:
                            pred_labels.append(p)
                    
                    pred_labels = np.reshape(pred_labels, [100,2500])
                    pred_labels = np.mean(pred_labels, axis = 0)

                    adj_matrix_true = truth[int(number)]

                    
                    pred_score = np.greater_equal(pred_labels,0.45) 
                    pred_score = pred_score.astype(int)
                    adj_matrix_predicted = np.reshape(pred_score, [50,50])
                    adj_matrix_mask = np.ones_like(adj_matrix_true)

                    v_mask = is_there_adjacency(adj_matrix_mask)
                    v_true = get_adjacency_type(adj_matrix_true) * v_mask
                    v_predicted = get_adjacency_type(adj_matrix_predicted) * v_mask
                    rec = ((v_true == v_predicted) & (v_true != 0)).sum() / (v_true != 0).sum()
                    pre = (
                        ((v_true == v_predicted) & (v_predicted != 0)).sum() / (v_predicted != 0).sum()
                        if (v_predicted != 0).sum() != 0
                        else 0.0
                    )
                    f1 = 2 * rec * pre / (pre + rec) if (rec + pre) != 0 else 0.0
                    logger.info("epochs {0}:   f1 {1:g}  pre {2:g}  rec {3:g}, loss {4:g}".format((iii +1), f1, pre, rec, np.mean(losses)))
                    best_saver1.handle(f1, sess, current_step)
                    

                    pred_score = np.greater_equal(pred_labels,0.455) 
                    pred_score = pred_score.astype(int)
                    adj_matrix_predicted = np.reshape(pred_score, [50,50])
                    adj_matrix_mask = np.ones_like(adj_matrix_true)

                    v_mask = is_there_adjacency(adj_matrix_mask)
                    v_true = get_adjacency_type(adj_matrix_true) * v_mask
                    v_predicted = get_adjacency_type(adj_matrix_predicted) * v_mask
                    rec = ((v_true == v_predicted) & (v_true != 0)).sum() / (v_true != 0).sum()
                    pre = (
                        ((v_true == v_predicted) & (v_predicted != 0)).sum() / (v_predicted != 0).sum()
                        if (v_predicted != 0).sum() != 0
                        else 0.0
                    )
                    f1 = 2 * rec * pre / (pre + rec) if (rec + pre) != 0 else 0.0
                    logger.info("epochs {0}:   f1 {1:g}  pre {2:g}  rec {3:g}".format((iii +1), f1, pre, rec))
                    
                    

                    pred_score = np.greater_equal(pred_labels,0.46) 
                    pred_score = pred_score.astype(int)
                    adj_matrix_predicted = np.reshape(pred_score, [50,50])
                    adj_matrix_mask = np.ones_like(adj_matrix_true)

                    v_mask = is_there_adjacency(adj_matrix_mask)
                    v_true = get_adjacency_type(adj_matrix_true) * v_mask
                    v_predicted = get_adjacency_type(adj_matrix_predicted) * v_mask
                    rec = ((v_true == v_predicted) & (v_true != 0)).sum() / (v_true != 0).sum()
                    pre = (
                        ((v_true == v_predicted) & (v_predicted != 0)).sum() / (v_predicted != 0).sum()
                        if (v_predicted != 0).sum() != 0
                        else 0.0
                    )
                    f1 = 2 * rec * pre / (pre + rec) if (rec + pre) != 0 else 0.0
                    logger.info("epochs {0}:   f1 {1:g}  pre {2:g}  rec {3:g}".format((iii +1), f1, pre, rec))
                    best_saver2.handle(f1, sess, current_step)

                    pred_score = np.greater_equal(pred_labels,0.465) 
                    pred_score = pred_score.astype(int)
                    adj_matrix_predicted = np.reshape(pred_score, [50,50])
                    adj_matrix_mask = np.ones_like(adj_matrix_true)

                    v_mask = is_there_adjacency(adj_matrix_mask)
                    v_true = get_adjacency_type(adj_matrix_true) * v_mask
                    v_predicted = get_adjacency_type(adj_matrix_predicted) * v_mask
                    rec = ((v_true == v_predicted) & (v_true != 0)).sum() / (v_true != 0).sum()
                    pre = (
                        ((v_true == v_predicted) & (v_predicted != 0)).sum() / (v_predicted != 0).sum()
                        if (v_predicted != 0).sum() != 0
                        else 0.0
                    )
                    f1 = 2 * rec * pre / (pre + rec) if (rec + pre) != 0 else 0.0
                    logger.info("epochs {0}:   f1 {1:g}  pre {2:g}  rec {3:g}, loss {4:g}".format((iii +1), f1, pre, rec, np.mean(losses)))

                    pred_score = np.greater_equal(pred_labels,0.47) 
                    pred_score = pred_score.astype(int)
                    adj_matrix_predicted = np.reshape(pred_score, [50,50])
                    adj_matrix_mask = np.ones_like(adj_matrix_true)

                    v_mask = is_there_adjacency(adj_matrix_mask)
                    v_true = get_adjacency_type(adj_matrix_true) * v_mask
                    v_predicted = get_adjacency_type(adj_matrix_predicted) * v_mask
                    rec = ((v_true == v_predicted) & (v_true != 0)).sum() / (v_true != 0).sum()
                    pre = (
                        ((v_true == v_predicted) & (v_predicted != 0)).sum() / (v_predicted != 0).sum()
                        if (v_predicted != 0).sum() != 0
                        else 0.0
                    )
                    f1 = 2 * rec * pre / (pre + rec) if (rec + pre) != 0 else 0.0
                    logger.info("epochs {0}:   f1 {1:g}  pre {2:g}  rec {3:g}, loss {4:g}".format((iii +1), f1, pre, rec, np.mean(losses)))

                    pred_score = np.greater_equal(pred_labels,0.448) 
                    pred_score = pred_score.astype(int)
                    adj_matrix_predicted = np.reshape(pred_score, [50,50])
                    adj_matrix_mask = np.ones_like(adj_matrix_true)

                    v_mask = is_there_adjacency(adj_matrix_mask)
                    v_true = get_adjacency_type(adj_matrix_true) * v_mask
                    v_predicted = get_adjacency_type(adj_matrix_predicted) * v_mask
                    rec = ((v_true == v_predicted) & (v_true != 0)).sum() / (v_true != 0).sum()
                    pre = (
                        ((v_true == v_predicted) & (v_predicted != 0)).sum() / (v_predicted != 0).sum()
                        if (v_predicted != 0).sum() != 0
                        else 0.0
                    )
                    f1 = 2 * rec * pre / (pre + rec) if (rec + pre) != 0 else 0.0
                    logger.info("epochs {0}:   f1 {1:g}  pre {2:g}  rec {3:g}, loss {4:g}".format((iii +1), f1, pre, rec, np.mean(losses)))

                    pred_score = np.greater_equal(pred_labels,0.445) 
                    pred_score = pred_score.astype(int)
                    adj_matrix_predicted = np.reshape(pred_score, [50,50])
                    adj_matrix_mask = np.ones_like(adj_matrix_true)

                    v_mask = is_there_adjacency(adj_matrix_mask)
                    v_true = get_adjacency_type(adj_matrix_true) * v_mask
                    v_predicted = get_adjacency_type(adj_matrix_predicted) * v_mask
                    rec = ((v_true == v_predicted) & (v_true != 0)).sum() / (v_true != 0).sum()
                    pre = (
                        ((v_true == v_predicted) & (v_predicted != 0)).sum() / (v_predicted != 0).sum()
                        if (v_predicted != 0).sum() != 0
                        else 0.0
                    )
                    f1 = 2 * rec * pre / (pre + rec) if (rec + pre) != 0 else 0.0
                    logger.info("epochs {0}:   f1 {1:g}  pre {2:g}  rec {3:g}, loss {4:g}".format((iii +1), f1, pre, rec, np.mean(losses)))


                    logger.info("VALIDATION {0}:    f1 {1:g}  pre {2:g}  rec {3:g}".format((iii +1)/FLAGS.evaluation_interval, f1, pre, rec))
                    
                    


                if ((iii+1) % FLAGS.checkpoint_every == 0):
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("Saved model checkpoint to {0}\n".format(path))

                logger.info("Epoch {0} has finished!".format(iii + 1))
            

    

    
    logger.info("Done.")


if __name__ == '__main__':
    train()
