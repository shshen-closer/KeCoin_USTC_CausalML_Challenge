# -*- coding:utf-8 -*-
__author__ = 'shshen'

import numpy as np
import tensorflow as tf

def weight_variable(shape,  name=None, training = None):
    initial = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling()(shape) , trainable=True, name=name)  
   # initial = tf.Variable(tf.random_uniform(shape, minval=-0.1, maxval=0.1,dtype=tf.float32), trainable=True, name=name)  
   # initial = tf.Variable(tf.random_normal(shape,mean=0.0,stddev=0.1,dtype=tf.float32), trainable=True, name=name)
    return initial
def bias_variable(shape,  name=None, training = None):
    initial = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling()(shape) , trainable=True, name=name) 
   # initial = tf.Variable(tf.random_uniform(shape, minval=-0.1, maxval=0.1,dtype=tf.float32), trainable=True, name=name)  
   # initial = tf.Variable(tf.random_normal(shape,mean=0.0,stddev=0.1,dtype=tf.float32), trainable=True, name=name)
    return initial


class LPKT(object):

    def __init__(self, batch_size, num_steps, num_skills, hidden_size):
        
        self.batch_size = batch_size = batch_size
        self.hidden_size  = hidden_size
        self.num_steps = num_steps
        self.num_skills =  num_skills

        self.input_problem = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps], name="input_problem")
        self.input_kc = tf.compat.v1.placeholder(tf.float32, [batch_size, num_steps, num_skills], name="input_kc")
        self.x_answer = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="x_answer")
        self.target_id = tf.compat.v1.placeholder(tf.int32, [batch_size,num_steps], name="target_id")
        self.target_kc = tf.compat.v1.placeholder(tf.float32, [batch_size, num_steps, num_skills], name="target_kc")
        self.target_index = tf.compat.v1.placeholder(tf.int32, [None], name="target_index")
        self.target_correctness = tf.compat.v1.placeholder(tf.float32, [None], name="target_correctness")

        
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
        
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.initializer=tf.compat.v1.keras.initializers.VarianceScaling()
        
        # exercise embedding
        self.problem_w = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling()([5730, hidden_size]),dtype=tf.float32, trainable=True, name = 'problem_w')
      #  zero_problem = tf.zeros((1, hidden_size))
      #  all_problem = tf.concat([zero_problem, self.problem_w ], axis = 0)

        problem_embedding =  tf.nn.embedding_lookup(self.problem_w, self.input_problem)
        target_id =  tf.nn.embedding_lookup(self.problem_w, self.target_id)

        print('problem_embedding', np.shape(problem_embedding))
        

        #answer embedding
        zeros = tf.zeros((1,50))
        ones =  tf.ones((1,50))
        ttt = tf.concat([zeros,ones],axis = 0)
        x_answer = tf.nn.embedding_lookup(ttt,self.x_answer)

        input_data = tf.concat([problem_embedding , x_answer],axis = -1)
        input_data = tf.compat.v1.layers.dense(input_data, units = hidden_size)  #* tf.nn.sigmoid(tf.compat.v1.layers.dense(input_data, units = hidden_size))
        print('input_data', np.shape(input_data))
      

        knowledge_matrix = tf.Variable(tf.compat.v1.keras.initializers.VarianceScaling()([num_skills, hidden_size]),dtype=tf.float32, trainable=True, name = 'knowledge_matrix')
        knowledge_matrix = tf.tile(tf.expand_dims(knowledge_matrix, 0), tf.stack([batch_size, 1, 1]))
      

        shape = input_data.get_shape().as_list()
        padd = tf.zeros((shape[0], 2, shape[2]))
        input_data = tf.concat([padd, input_data], axis = 1)
        slice_input_data = tf.split(input_data, self.num_steps + 2, 1)

        padd = tf.zeros((batch_size, 2, num_skills))
        input_kc = tf.concat([padd, self.input_kc], axis = 1)
        slice_input_kc = tf.split(input_kc, self.num_steps + 2, 1)
        
        print(shape)

        h = list()



        w_f = weight_variable([shape[2],  2*hidden_size], name = 'w_f', training = self.is_training )
        b_f = bias_variable([shape[2]],  name='b_f', training = self.is_training )

        w_l = weight_variable([shape[2], 3*shape[2]], name = 'w_l', training = self.is_training )
        b_l = bias_variable([shape[2]],  name='b_l', training = self.is_training )
        
        w_o = weight_variable([shape[2], 1*hidden_size], name = 'w_o', training = self.is_training )
        b_o = bias_variable([shape[2]],  name='b_o', training = self.is_training )

        w_c = weight_variable([shape[2], 3*shape[2]], name = 'w_c', training = self.is_training )
        b_c = bias_variable([shape[2]],  name='b_c', training = self.is_training )
        
        reuse_flag = False
        for i in range(2,self.num_steps+2):
            if i != 0:
                reuse_flag = True

            kc_one = slice_input_kc[i]

            q1 = tf.squeeze(slice_input_data[i], 1)  # b, hidden_size
            q0 = tf.squeeze(slice_input_data[i-1], 1)  # b, hidden_size
            q_1 = tf.squeeze(slice_input_data[i-2], 1)  # b, hidden_size
            kkk = tf.matmul(kc_one, knowledge_matrix)
            kkk = tf.squeeze(kkk, 1)

            q = tf.concat([q0, q1, kkk], axis = -1)
            att_score = kc_one  # b, 1, num_skills

            

            learn_gates = tf.sigmoid(tf.matmul(q,  tf.transpose(w_l, [1,0])+b_l), name='l_gate')
           
            c_title = tf.tanh(tf.matmul(q,  tf.transpose(w_c, [1,0])+b_c), name='c_gate')
          #  c_title = tf.nn.dropout(c_title, self.dropout_keep_prob)
            learn_gates = learn_gates * (1+c_title)/2
           # learn_gates = tf.nn.dropout(learn_gates, self.dropout_keep_prob)
            learn_gains = tf.expand_dims(learn_gates, axis = -1)
            learn_gains = tf.matmul(learn_gains, att_score)
            learn_gains = tf.transpose(learn_gains, [0,2,1])
          #  learn_gains = tf.nn.dropout(learn_gains, self.dropout_keep_prob)

            learn_gates_t = tf.expand_dims(learn_gates, axis = 1)
            learn_gates_t = tf.tile(learn_gates_t, [1,num_skills,1])

            q_f =  tf.concat([knowledge_matrix, learn_gates_t], axis = -1)
            for_get = tf.matmul(q_f, tf.transpose(w_f, [1,0])+b_f)
            for_get = tf.sigmoid(for_get)



            knowledge_matrix = learn_gains  + knowledge_matrix * for_get 
            h_i = tf.expand_dims(knowledge_matrix, axis = 1)
            h.append(h_i)

        print('hh',np.shape(h))
        output = tf.concat(h, axis = 1)
        print('output',np.shape(output))  #b, 100, 265, hidden_size
        target_kc = tf.expand_dims(self.target_kc, axis = 2)
   
        output = tf.matmul(target_kc, output) # b,500,1,hidden_size
        output = tf.reduce_mean(output, axis = 2)
        output = tf.concat([target_id,output], axis = -1)
        output = tf.compat.v1.layers.dense(output, units = hidden_size)
        logits = tf.reduce_mean(output, axis = -1, name="logits")
      
        print('logits',np.shape(logits))
        logits = tf.reshape(logits, [-1])
        self.preds = tf.sigmoid(logits, name="preds")

        selected_logits = tf.gather(logits, self.target_index)
        
        #make prediction
        self.pred = tf.sigmoid(selected_logits, name="pred")

        # loss function
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=selected_logits, labels=self.target_correctness), name="losses")

        self.cost = self.loss