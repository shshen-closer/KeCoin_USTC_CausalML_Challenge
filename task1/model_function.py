# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:14:05 2019

@author:shshen
"""

import tensorflow as tf
import numpy as np






def cnn_block(x, filters, kk,  is_training, drop_rate):
    
    o1 = tf.nn.swish(tf.compat.v1.layers.batch_normalization(x, training=is_training))
    
    res_w = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling()([kk, filters, filters]),dtype=tf.float32)

    res_b = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling()([filters]),dtype=tf.float32)
    
    o2 = tf.nn.conv1d(o1, res_w, 1, padding='SAME') + res_b

    

    return o2 



def multi_span(x, hidden_size, is_training, drop_rate):


    for iii in range(9, 0, -1):
      
      
      x = cnn_block(x,hidden_size, iii,  is_training, drop_rate)
      x = cnn_block(x,hidden_size, 3,  is_training, drop_rate)
     # x = tf.compat.v1.layers.average_pooling1d(x, pool_size = 2, strides = 2, padding='SAME')
  
    print('x after conv: ', np.shape(x))
 

   # x = tf.compat.v1.layers.dense(x1, 2*hidden_size)
    x = tf.compat.v1.layers.dense(x, hidden_size)
    x = tf.reduce_mean(x, axis = 1)
    
    
    print(np.shape(x))
    return x
 
