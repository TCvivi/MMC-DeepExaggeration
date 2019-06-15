# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
import tensorflow as tf
import rnn

def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return tf.contrib.training.HParams(**hparams.values())

def get_default_hparams():
    """Return default HParams for sketch-rnn."""
    hparams = tf.contrib.training.HParams(
        source = '1people5.npy',  
        target = '1cartoon5.npy', 
        sourcesTest = '1newsources5.npy',  
        targetsTest = '1newtargets5.npy',
        num_steps=10000,  # Total number of steps of training. Keep large. 迭代次数
        save_every=200,  # Number of batches per checkpoint creation.
        max_seq_len=68,  # Not used. Will be changed by model. [Eliminate?] 最长的序列长度
        dec_rnn_size=512,  # Size of decoder. 解码器隐藏层单元数目
        dec_model='lstm',  # Decoder: lstm, layer_norm or hyper.
        enc_rnn_size=256,  # Size of encoder.编码器隐藏层单元数目
        enc_model='lstm',  # Encoder: lstm, layer_norm or hyper.
        z_size=128,  # Size of latent vector z. Recommend 32, 64 or 128. 隐含变量维度
        batch_size=64,  # Minibatch size. Recommend leaving at 100. 
        grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
        num_mixture=10,  # Number of mixtures in Gaussian mixture model. 高斯混合模型的个数
        learning_rate=0.0001,  # Learning rate. 学习率
        decay_rate=0.9999,  # Learning rate decay per minibatch.
        min_learning_rate=0.00001,  # Minimum learning rate. 最小的学习率
        use_recurrent_dropout=True,  # Dropout with memory loss. Recomended
        recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
        use_input_dropout=False,  # Input dropout. Recommend leaving False.
        input_dropout_prob=0.90,  # Probability of input dropout keep.
        use_output_dropout=False,  # Output droput. Recommend leaving False.
        output_dropout_prob=0.90,  # Probability of output dropout keep.
        random_scale_factor=0.15,  # 相关参数的初始值为随机均匀分布
        conditional=True,  # When False, use unconditional decoder-only model.
        is_training=True  # Is model training? Recommend keeping true.
     )
    return hparams

# 创建SketchRNN模型
class Model(object):
    def __init__(self, hps, gpu_mode=True, reuse=False):
        tf.logging.info('model-创建seq2seqmodel====================================')
        self.hps = hps #网络参数
        with tf.variable_scope('vector_rnn', reuse=reuse):
            if not gpu_mode: #使用gpu
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using cpu.')
                    self.build_model(hps)
            else:
                tf.logging.info('Model using gpu.')
                self.build_model(hps) 
                #构建一个以hps为参数的模型
    
    #创建一个双向的encoder 输入batch和sequece的长度
    def encoder(self, batch, sequence_lengths): 
        tf.logging.info('model-encoder部分')
        unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
            self.enc_cell_fw,
            self.enc_cell_bw,
            batch,
            sequence_length=sequence_lengths,
            time_major=False,
            swap_memory=True,
            dtype=tf.float32,
            scope='ENC_RNN')

        last_state_fw, last_state_bw = last_states
        last_h_fw = self.enc_cell_fw.get_output(last_state_fw) #得到正向RNN的最后的隐层输出
        last_h_bw = self.enc_cell_bw.get_output(last_state_bw) #得到反向RNN的最后的隐层输出
        last_h = tf.concat([last_h_fw, last_h_bw], 1) #bi-direction rnn 的最后隐层结果 h
        latent_h = rnn.super_linear(
            last_h,
            self.hps.z_size, #output size
            input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
            scope='latent_h',
            init_w='gaussian',
            weight_start=0.001)
        return latent_h #128维的encoder最终输出

    #构建模型结构
    def build_model(self, hps): 
        if hps.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if hps.dec_model == 'lstm':
            cell_fn = rnn.LSTMCell  
        elif hps.dec_model == 'layer_norm':
            cell_fn = rnn.LayerNormLSTMCell
        elif hps.dec_model == 'hyper':
            cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        if hps.enc_model == 'lstm':
            enc_cell_fn = rnn.LSTMCell #设置编码器为LSTM
        elif hps.enc_model == 'layer_norm':
            enc_cell_fn = rnn.LayerNormLSTMCell
        elif hps.enc_model == 'hyper':
            enc_cell_fn = rnn.HyperLSTMCell
        else:
            assert False, 'please choose a respectable cell'

        #dropout技巧
        use_recurrent_dropout = self.hps.use_recurrent_dropout
        use_input_dropout = self.hps.use_input_dropout
        use_output_dropout = self.hps.use_output_dropout

        cell = cell_fn(
            hps.dec_rnn_size, #512
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

        if hps.conditional:  
            if hps.enc_model == 'hyper':
                self.enc_cell_fw = enc_cell_fn(hps.enc_rnn_size,
                            use_recurrent_dropout=use_recurrent_dropout,
                            dropout_keep_prob=self.hps.recurrent_dropout_prob)
                self.enc_cell_bw = enc_cell_fn(hps.enc_rnn_size,
                            use_recurrent_dropout=use_recurrent_dropout,
                            dropout_keep_prob=self.hps.recurrent_dropout_prob)
            else: #enc_model = lstm
                self.enc_cell_fw = enc_cell_fn( #双向RNN的正向
                            hps.enc_rnn_size,
                            use_recurrent_dropout=use_recurrent_dropout,
                            dropout_keep_prob=self.hps.recurrent_dropout_prob)
                self.enc_cell_bw = enc_cell_fn( #双向RNN的反向
                            hps.enc_rnn_size,
                            use_recurrent_dropout=use_recurrent_dropout,
                            dropout_keep_prob=self.hps.recurrent_dropout_prob)

        # dropout:
        tf.logging.info('Input dropout mode = %s.', use_input_dropout) #false
        tf.logging.info('Output dropout mode = %s.', use_output_dropout) #false
        tf.logging.info('Recurrent dropout mode = %s.', use_recurrent_dropout) #true
        if use_input_dropout:
            tf.logging.info('Dropout to input w/ keep_prob = %4.4f.',
                      self.hps.input_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, 
                                    input_keep_prob=self.hps.input_dropout_prob)
        if use_output_dropout:
            tf.logging.info('Dropout to output w/ keep_prob = %4.4f.',
                      self.hps.output_dropout_prob)
            cell = tf.contrib.rnn.DropoutWrapper(cell, 
                                    output_keep_prob=self.hps.output_dropout_prob)
        
        #==============================decode !!!============================
        tf.logging.info('model- decoder 部分')
        self.cell = cell
        self.sequence_lengths = tf.placeholder(
            dtype=tf.int32, shape=[self.hps.batch_size]) #batch 大小        
        #包含了起始符的decoder输入 
        self.source_input = tf.placeholder(dtype=tf.float32,
                       shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5])
        self.target_input = tf.placeholder(dtype=tf.float32,
                       shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5]) 
        
        tf.logging.info('model- encoder的输入')
        self.encoder_input_x = self.source_input[:, 1:self.hps.max_seq_len + 1, :] 
        tf.logging.info('model- decoder的输入和标签')
        self.output_x = self.target_input[:, 1:self.hps.max_seq_len + 1, :]         
        self.decoder_input_source_x = self.source_input[:, :self.hps.max_seq_len, :]        
        self.decoder_input_target_x = self.target_input[:, :self.hps.max_seq_len, :]

        # 如果condition=true，输入加入隐含变量z
        if hps.conditional:  
            self.laten = self.encoder(self.encoder_input_x, self.sequence_lengths) 
            pre_tile_y = tf.reshape(self.laten, [self.hps.batch_size, 1, self.hps.z_size]) 
            overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])
            #输入数据，得到隐含变量h            
            actual_input_x = tf.concat([self.decoder_input_target_x, overlay_x], 2) 
            # decoder每一时刻的输入为 h和 target输入的组合 按咧拼接’      
            self.initial_state = tf.nn.tanh(rnn.super_linear(self.laten,
                              cell.state_size,init_w='gaussian',weight_start=0.001,
                              input_size=self.hps.z_size))
        # unconditional, decoder-only generation
        else:  
            self.laten = tf.zeros((self.hps.batch_size, self.hps.z_size), dtype=tf.float32)
            actual_input_x = self.input_x
            self.initial_state = cell.zero_state(batch_size=hps.batch_size, dtype=tf.float32)

        tf.logging.info('model- 开始高斯混合模型采样了======================================')
        self.num_mixture = hps.num_mixture #混合高斯模型的高斯个数 20
        n_out = (3 + self.num_mixture * 6) 
        #解码器输出y的维度为 5M + M + 3，分别表示高斯函数参数、权重、（p1,p2,p3）
        with tf.variable_scope('RNN'):
            output_w = tf.get_variable('output_w', [self.hps.dec_rnn_size, n_out])
            output_b = tf.get_variable('output_b', [n_out])

        # decoder module of sketch-rnn is below
        output, last_state = tf.nn.dynamic_rnn( cell, actual_input_x,
                        initial_state=self.initial_state,time_major=False,
                        swap_memory=True,dtype=tf.float32,scope='RNN')

        output = tf.reshape(output, [-1, hps.dec_rnn_size])
        output = tf.nn.xw_plus_b(output, output_w, output_b) #全连接层，接n_out个神经元
        self.final_state = last_state

        # x1 x2分别是坐标x轴和y轴的偏移量，result为二维正态分布的概率密度
        def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
            tf.logging.info('model- 根据那篇文章的公式计算二维正态分布的概率密度')
            norm1 = tf.subtract(x1, mu1)
            norm2 = tf.subtract(x2, mu2)
            s1s2 = tf.multiply(s1, s2)
            z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -2 *
                    tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
            neg_rho = 1 - tf.square(rho)
            result = tf.exp(tf.div(-z, 2 * neg_rho))
            denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
            result = tf.div(result, denom)
            return result        
                
        #计算重构误差，求上面概率密度的对数
        def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                     z_pen_logits, x1_data, x2_data, pen_data):
            tf.logging.info('model- 计算重构误差')
            #采样（x,y）数据的误差        
            result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
            epsilon = 1e-6
            result1 = tf.multiply(result0, z_pi) #每一个高斯模型乘上相应的权重
            result1 = tf.reduce_sum(result1, 1, keep_dims=True) #对所有求和
            result1 = -tf.log(result1 + epsilon)  # avoid log(0) 
            fs = 1.0 - pen_data[:,2]  #如果最后一个点为1，则为终点概率为0，反之，为1. 
            fs = tf.reshape(fs, [-1, 1]) #将batch * 1 的数据转为 1* batch的数据
            result1 = tf.multiply(result1, fs)

            # result2: loss wrt pen state, (L_p in equation 9)
            result2 = tf.nn.softmax_cross_entropy_with_logits( #Lp就是求交叉熵
                      labels=pen_data, logits=z_pen_logits)
            result2 = tf.reshape(result2, [-1, 1])
            if not self.hps.is_training:  # eval mode, mask eos columns
                result2 = tf.multiply(result2, fs)

            result = result1 + result2
            return result

        # below is where we need to do MDN (Mixture Density Network) splitting of
        # distribution params
        def get_mixture_coef(output):
            tf.logging.info('model- 将decoder的网络输出切分成高斯混合模型的参数')
            """Returns the tf slices containing mdn dist params."""
            #根据decoder输出的6M+8的值，构建混合高斯模型
            z = output
            z_pen_logits = z[:, 0:3]  # pen states z的前三个值为（p1,p2,p3）
            #剩下的6个分别为权重pi，miu(x),miu(y), z_sigma1, z_sigma2, z_corr
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

            # process output z's into MDN paramters

            # softmax all the pi's and pen states:
            z_pi = tf.nn.softmax(z_pi)
            z_pen = tf.nn.softmax(z_pen_logits) 
            #对(p1,p2,p3)和pi的值做做logit处理，是的其都为正，且加起来为1

            # exponentiate the sigmas and also make corr between -1 and 1.
            z_sigma1 = tf.exp(z_sigma1)
            z_sigma2 = tf.exp(z_sigma2)
            z_corr = tf.tanh(z_corr)

            r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
            return r 

        tf.logging.info('model- 调用切分输出函数')
        out = get_mixture_coef(output)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

        self.pi = o_pi
        self.mu1 = o_mu1
        self.mu2 = o_mu2
        self.sigma1 = o_sigma1
        self.sigma2 = o_sigma2
        self.corr = o_corr
        self.pen_logits = o_pen_logits
        self.pen = o_pen

        # reshape target data so that it is compatible with prediction shape
        target = tf.reshape(self.output_x, [-1, 5]) #目标输出（x,y ,p1,p2,p3）
        [x1_data, x2_data, p1, p2, p3] = tf.split(target, 5, 1)
        pen_data = tf.concat([p1, p2, p3], 1) 

        lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,
                            o_pen_logits, x1_data, x2_data, pen_data)

        self.r_cost = tf.reduce_mean(lossfunc) #求重构误差
        if self.hps.is_training:
            tf.logging.info('model- 选择学习率和误差函数')
            self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr) #使用ADAM优化方式           
            self.cost = self.r_cost 
            gvs = optimizer.compute_gradients(self.cost)
            g = self.hps.grad_clip # 1.0
            capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(
                      capped_gvs, global_step=self.global_step, name='train_step')