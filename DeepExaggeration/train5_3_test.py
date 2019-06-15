# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import time
import urllib
import zipfile
import numpy as np
import requests
import six
from six.moves import cStringIO as StringIO
import tensorflow as tf
import model5_3_KL_R_C as sketch_rnn_model
import utils

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS

# tf.flags 定义全局参数设置
tf.app.flags.DEFINE_string('data_dir', '../newFaceAlignment/data_new/output_npy/','the dataset filepath')
tf.app.flags.DEFINE_string('log_root', 'parameters_5_3','Directory to store model')
tf.app.flags.DEFINE_boolean('resume_training', False,'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string('hparams', '','Pass in comma-separated key=value pairs')

def reset_graph():
    """Closes the current default session and resets the graph."""
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()


def load_env(data_dir, model_dir):
    """Loads environment for inference mode, used in jupyter notebook."""
    model_params = sketch_rnn_model.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_params.parse_json(f.read())
    return load_dataset(data_dir, model_params, inference_mode=True)


def load_model(model_dir):
    """Loads model for inference mode, used in jupyter notebook."""
    model_params = sketch_rnn_model.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        model_params.parse_json(f.read())

    model_params.batch_size = 1  # only sample one at a time
    eval_model_params = sketch_rnn_model.copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 0
    sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
    sample_model_params.max_seq_len = 1  # sample one point at a time
    return [model_params, eval_model_params, sample_model_params]

def load_dataset(data_dir, model_params, inference_mode=False): 
    tf.logging.info('loaddataset-开始数据处理=================================================')
    #训练集
    source = model_params.source #source.npy   
    target = model_params.target #target.npy
    source_data = np.load(os.path.join(data_dir, source))
    target_data = np.load(os.path.join(data_dir, target))
    tf.logging.info('打印原始输入长度 %i.',len(source_data))
    tf.logging.info('打印目标输入长度 %i.',len(target_data))
    
    #测试集
    sourcesTest = model_params.sourcesTest  
    targetsTest = model_params.targetsTest 
    source_test_data = np.load(os.path.join(data_dir, sourcesTest))
    target_test_data = np.load(os.path.join(data_dir, targetsTest)) 
    tf.logging.info('打印测试集原始输入长度 %i.',len(source_test_data))
    tf.logging.info('打印测试集目标输入长度 %i.',len(target_test_data))
    
    num_points = 68
    model_params.max_seq_len = num_points 
    tf.logging.info('model_params.max_seq_len %i.', model_params.max_seq_len) #并打印出来
    
    eval_model_params = sketch_rnn_model.copy_hparams(model_params) #讲model的参数复制给评价模型
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0 #并修改一些参数
    eval_model_params.use_output_dropout = 0
    eval_model_params.is_training = 1

    if inference_mode: # = fales
        eval_model_params.batch_size = 1
        eval_model_params.is_training = 0

    sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params) #将参数复制给sample模型
    sample_model_params.batch_size = 1  # only sample one at a time
    sample_model_params.max_seq_len = 1  # sample one point at a time

    #随机打乱后依次取出一个batch的训练集、测试集、验证集数据
    #数据的x,y做过normalize处理，且不足Nmax的补充为(0,0,0,0,1)
    tf.logging.info('正式处理数据，输入网络')
    #为保证source和target同顺序
    indices = np.random.permutation(range(0, len(source_data)))[0:model_params.batch_size] 
    
    #训练集
    source_set = utils.DataLoader( source_data, indices,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor)
    target_set = utils.DataLoader( target_data, indices,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor)
    
    factor_source = source_set.calculate_normalizing_scale_factor() 
    source_set.normalize(factor_source)#再对数据做normalize
    factor_target = target_set.calculate_normalizing_scale_factor() 
    target_set.normalize(factor_target)#再对数据做normalize
    
    #测试集
    source_test_set = utils.DataLoader(source_test_data, indices,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor)
    target_test_set = utils.DataLoader(target_test_data, indices,
        model_params.batch_size,
        max_seq_length=model_params.max_seq_len,
        random_scale_factor=model_params.random_scale_factor)
    
    factor_source_test = source_test_set.calculate_normalizing_scale_factor() 
    source_test_set.normalize(factor_source_test)#再对数据做normalize
    factor_target_test = target_test_set.calculate_normalizing_scale_factor() 
    target_test_set.normalize(factor_target_test)#再对数据做normalize

    tf.logging.info('source normalizing_scale_factor is %4.4f.',factor_source) 
    tf.logging.info('target normalizing_scale_factor is %4.4f.',factor_target)

    result = [source_set, target_set, source_test_set, target_test_set,
              model_params, eval_model_params, sample_model_params]
    return result

def evaluate_model(sess, model, sources, targets):
    """Returns the average weighted cost, reconstruction cost and KL cost."""
    total_cost = 0.0
    total_r_cost = 0.0
    total_kl_cost = 0.0
    total_i_cost = 0.0
    for batch in range(sources.num_batches): 
        start_idx = batch * 64
        indices = range(start_idx, start_idx + 64)
        unused_orig_x, x, s = sources._get_batch_from_indices(indices)
        unused_orig_y, y, ss = targets._get_batch_from_indices(indices)
        feed = {model.source_input: x,
              model.target_input: y, 
              model.sequence_lengths: s}
        
        (cost, r_cost, kl_cost, i_cost) = sess.run([model.cost, model.r_cost, model.kl_cost, model.i_cost], feed)
        total_cost += cost
        total_r_cost += r_cost
        total_kl_cost += kl_cost
        total_i_cost += i_cost

    total_cost /= (sources.num_batches)
    total_r_cost /= (sources.num_batches)
    total_kl_cost /= (sources.num_batches)
    total_i_cost /= (sources.num_batches)
    return (total_cost, total_r_cost, total_kl_cost, total_i_cost)

def load_checkpoint(sess, checkpoint_path):
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

def save_model(sess, model_save_path, global_step):
    saver = tf.train.Saver(tf.global_variables())
    tf.logging.info('global_step %i saving model  in %s', global_step, model_save_path)
    checkpoint_path = os.path.join(model_save_path, str(global_step))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    saver.save(sess, checkpoint_path, global_step=global_step)

def train(sess, model, eval_model, source_set, target_set):
    """Train a sketch-rnn model."""
    # Setup summary writer.
    summary_writer = tf.summary.FileWriter(FLAGS.log_root)

    # Calculate trainable params.
    t_vars = tf.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        tf.logging.info('%s %s %i', var.name, str(var.get_shape()), num_param)
    # 打印网络参数
    tf.logging.info('train- Total trainable variables %i.', count_t_vars) #1901824
    model_summ = tf.summary.Summary()
    model_summ.value.add(
        tag='Num_Trainable_Params', simple_value=float(count_t_vars))
    summary_writer.add_summary(model_summ, 0)
    summary_writer.flush()

    # setup eval stats
    best_test_cost = 100000000.0  # set a large init value
    test_cost = 0.0

    # main train loop
    hps = model.hps
    start = time.time()
    for _ in range(hps.num_steps):
        step = sess.run(model.global_step)
        curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                          (hps.decay_rate)**step + hps.min_learning_rate)
        curr_kl_weight = (hps.kl_weight - (hps.kl_weight - hps.kl_weight_start) *
                      (hps.kl_decay_rate)**step)
        curr_il_weight = (hps.il_weight - (hps.il_weight - hps.il_weight_start) *
                      (hps.il_decay_rate)**step)

        idx = np.random.permutation(range(0, 475))[0:64] 
        #数据集数目、batch大小改了要来改这里！！！！
        _, x, s1 = source_set._get_batch_from_indices(idx)
        _, y, s2 = target_set._get_batch_from_indices(idx)
 
        feed = {
            model.source_input: x,
            model.target_input: y,
            model.sequence_lengths: s1,
            model.lr: curr_learning_rate,
            model.kl_weight: curr_kl_weight,
            model.il_weight: curr_il_weight
        }
        
        (train_cost, r_cost, kl_cost, i_cost, _, train_step, _) = sess.run([
            model.cost, model.r_cost, model.kl_cost, model.i_cost, model.final_state,
            model.global_step, model.train_op], feed) #这里大大的问题

        if step % 10 == 0 and step > 0: #每10个step打印一次结果
            end = time.time()
            time_taken = end - start

            cost_summ = tf.summary.Summary()
            cost_summ.value.add(tag='Train_Cost', simple_value=float(train_cost)) #总误差
            reconstr_summ = tf.summary.Summary()
            reconstr_summ.value.add(
                tag='Train_Reconstr_Cost', simple_value=float(r_cost)) #重构误差
            kl_summ = tf.summary.Summary()
            kl_summ.value.add(tag='Train_KL_Cost', simple_value=float(kl_cost)) #KL散度
            identify_summ = tf.summary.Summary()
            identify_summ.value.add(tag='Train_identify_Cost', simple_value=float(i_cost)) 
            lr_summ = tf.summary.Summary()
            lr_summ.value.add(
                tag='Learning_Rate', simple_value=float(curr_learning_rate)) 
            kl_weight_summ = tf.summary.Summary()
            kl_weight_summ.value.add(tag='KL_Weight', simple_value=float(curr_kl_weight))
            il_weight_summ = tf.summary.Summary()
            il_weight_summ.value.add(tag='IL_Weight', simple_value=float(curr_il_weight))
            time_summ = tf.summary.Summary()
            time_summ.value.add(
                tag='Time_Taken_Train', simple_value=float(time_taken))

            output_format = ('step: %d, lr: %.6f, klw: %0.4f, cost: %.4f, '
                       'recon: %.4f, kl: %.4f, identify: %.4f, train_time_taken: %.4f')
            output_values = (step, curr_learning_rate, curr_kl_weight, 
                             train_cost, r_cost, kl_cost, i_cost, time_taken)
            output_log = output_format % output_values

            tf.logging.info(output_log)

            summary_writer.add_summary(cost_summ, train_step)
            summary_writer.add_summary(reconstr_summ, train_step)
            summary_writer.add_summary(kl_summ, train_step)
            summary_writer.add_summary(identify_summ, train_step)
            summary_writer.add_summary(lr_summ, train_step)
            summary_writer.add_summary(kl_weight_summ, train_step)
            summary_writer.add_summary(il_weight_summ, train_step)
            summary_writer.add_summary(time_summ, train_step)
            summary_writer.flush()
            start = time.time()
         
        if step % hps.save_every == 0 and step > 0:
            (test_cost, test_r_cost, test_kl_cost, test_i_cost) = evaluate_model(
                           sess, eval_model, source_set, target_set)
            end = time.time()
            time_taken_test = end - start
            start = time.time()
            test_cost_summ = tf.summary.Summary()
            test_cost_summ.value.add(tag='Test_Cost', simple_value=float(test_cost))
            test_reconstr_summ = tf.summary.Summary()
            test_reconstr_summ.value.add(tag='test_Reconstr_Cost', 
                                simple_value=float(test_r_cost))
            test_kl_summ = tf.summary.Summary()
            test_kl_summ.value.add(tag='test_KL_Cost', simple_value=float(test_kl_cost))
            test_identify_summ = tf.summary.Summary()
            test_identify_summ.value.add(tag='test_Identify_Cost', 
                                simple_value=float(test_i_cost))
            test_time_summ = tf.summary.Summary()
            test_time_summ.value.add(tag='Time_Taken_test', 
                                simple_value=float(time_taken_test))

            output_format = ('best_test_cost: %0.4f, test_cost: %.4f, test_recon: '
                       '%.4f, test_kl: %.4f, test_i: %.4f, test_time_taken: %.4f')
            output_values = (min(best_test_cost, test_cost), test_cost,
                       test_r_cost, test_kl_cost, test_i_cost, time_taken_test)
            output_log = output_format % output_values
            tf.logging.info(output_log)

            summary_writer.add_summary(test_cost_summ, train_step)
            summary_writer.add_summary(test_reconstr_summ, train_step)
            summary_writer.add_summary(test_kl_summ, train_step)
            summary_writer.add_summary(test_identify_summ, train_step)
            summary_writer.add_summary(test_time_summ, train_step)
            summary_writer.flush()
            
            if test_cost < best_test_cost:
                best_test_cost = test_cost
                save_model(sess, FLAGS.log_root, step) 
                end = time.time()
                time_taken_save = end - start
                start = time.time()
                tf.logging.info('time_taken_save %4.4f.', time_taken_save)
                best_test_cost_summ = tf.summary.Summary()
                best_test_cost_summ.value.add(tag='Best_Test_Cost', 
                                simple_value=float(best_test_cost))
                summary_writer.add_summary(best_test_cost_summ, train_step)
                summary_writer.flush()        
    
def trainer(model_params):
    """Train a sketch-rnn model."""
    np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)
    # 避免ndarray数据维度过大导致的输出不全

    #打印模型参数
    tf.logging.info('sketch-rnn')
    tf.logging.info('Hyperparams:')
    for key, val in six.iteritems(model_params.values()):
        tf.logging.info('%s = %s', key, str(val)) #将model的参数按照 key = value的格式打印出来
    
     #加载数据 result = [train_set, model_params, eval_model_params,sample_model_params]
    tf.logging.info('train-加载数据')
    datasets = load_dataset(FLAGS.data_dir, model_params) #按btach大小加载数据，且对数据做预处理
    source_set = datasets[0]
    target_set = datasets[1] 
    source_test_set = datasets[2]
    target_test_set = datasets[3]
    model_params = datasets[4]
    eval_model_params = datasets[5]

    reset_graph()
    tf.logging.info('train-进入model环节===================================================')
    model = sketch_rnn_model.Model(model_params) #构建网络模型确定技巧，根据输入计算cost并优化
    eval_model = sketch_rnn_model.Model(eval_model_params, reuse=True)

    tf.logging.info('train-分配显存')
    gpu_fraction = 0.1
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    if FLAGS.resume_training:
        load_checkpoint(sess, FLAGS.log_root)

    # Write config file to json file.
    tf.gfile.MakeDirs(FLAGS.log_root) #这个地址找不到？？？
    with tf.gfile.Open(
        os.path.join(FLAGS.log_root, 'model_config.json'), 'w') as f:
        json.dump(model_params.values(), f, indent=True)

    tf.logging.info('train- 开始进入train过程==========================================')
    train(sess, model, eval_model, source_set, target_set)


def main(unused_argv):
    tf.logging.info('train-main-导入模型参数=======================================')
    """Load model params, save config file and start trainer."""
    model_params = sketch_rnn_model.get_default_hparams() #得到网络参数
    if FLAGS.hparams:
        model_params.parse(FLAGS.hparams) #将参数写成 key = value的格式 
    tf.logging.info('train-main-开始trainer')
    trainer(model_params)


def console_entry_point():
    tf.logging.info('train- console_entry_point - 执行main函数')
    tf.app.run(main)


if __name__ == '__main__':
    tf.logging.info('train-开始啦')
    console_entry_point()
    
