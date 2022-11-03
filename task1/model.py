# -*- coding:utf-8 -*-
__author__ = 'shshen'

import numpy as np
import tensorflow as tf
from model_function import *



class model1(object):

    def __init__(self, batch_size, hidden_size, leng):
        

        self.construct1 = tf.compat.v1.placeholder(tf.int32, [batch_size, leng], name="construct1")
        self.construct2 = tf.compat.v1.placeholder(tf.int32, [batch_size, leng], name="construct2")
        self.construct3 = tf.compat.v1.placeholder(tf.int32, [batch_size, leng], name="construct3")
        self.construct4 = tf.compat.v1.placeholder(tf.int32, [batch_size, leng], name="construct4")
        self.target_relation = tf.compat.v1.placeholder(tf.float32, [batch_size], name="target_relation")

        
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
        
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.initializer=tf.compat.v1.keras.initializers.glorot_uniform() #keras.initializers.VarianceScaling()   random_normal_initializer
        


        self.embedding = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([1000, hidden_size]),dtype=tf.float32, trainable=True, name = 'embedding')
        c1 =  tf.nn.embedding_lookup(self.embedding, self.construct1) #batch_size, 400, hidden_size 
        c2 =  tf.nn.embedding_lookup(self.embedding, self.construct2) #batch_size, 400, hidden_size 

        self.embedding2 = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform()([3, hidden_size]),dtype=tf.float32, trainable=True, name = 'embedding2')
        c3 =  tf.nn.embedding_lookup(self.embedding2, self.construct3) #batch_size, 400, hidden_size 
        c4 =  tf.nn.embedding_lookup(self.embedding2, self.construct4) #batch_size, 400, hidden_size 



        c1 =  tf.concat([c1, c3, c1+c3, c1-c3, c1*c3], axis = -1) #current_Q * self.previous_knowledge #
        c1 = tf.compat.v1.layers.dense(c1, units = hidden_size)
        c2 =  tf.concat([c2, c4, c4+c2, c2-c4, c4*c2], axis = -1) #current_Q * self.previous_knowledge #
        c2 = tf.compat.v1.layers.dense(c2, units = hidden_size)


        inputs =  tf.concat([c1, c2, c1+c2, c1-c2, c1*c2], axis = -1) #current_Q * self.previous_knowledge #

        inputs = tf.compat.v1.layers.dense(inputs, units = hidden_size)

        outputs = multi_span(inputs, hidden_size, self.is_training, self.dropout_keep_prob)

        outputs = tf.compat.v1.layers.dense(outputs, units = hidden_size)
        
        self.logits = tf.compat.v1.layers.dense(outputs, units = 1)
        

        
        logits = tf.reshape(self.logits, [-1])
        
        #make prediction
        self.pred = tf.sigmoid(logits, name="pred")

        
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.target_relation), name="losses") 

        
        self.cost = self.loss

