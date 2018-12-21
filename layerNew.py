#!/usr/bin/python
#coding:utf-8
import os
import sys
import tensorflow as tf

"""
模块说明：
    对各种layer进行封装
函数说明：
    ### Layers
    1）初始化函数
      def __init__(self): 
    2）获取图像信息
      def get_img_info(image_path):10000):
    3）读取TFrecords文件
      def read_and_decode(tfrecord_files,image_height,image_width,image_mode,is_shuffle=True):
    ### Helpers
    4）创建每一层的权重weights
      def _weights(name, shape, initializer='normal',mean=0.0, stddev=0.02)
      注意：出于实际工作需要，另外追加了xavier初始化方式，如果需要实现其它初始化方式请在该函数中自行添加
    5）创建每一层的偏置biases
      def _biases(name, shape, constant=0.0):
    6）正则化函数，实现了instance normalization和batch_normal
      def _norm(input, is_training, norm='instance'):
   
"""

### Layers
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         initializer='xavier',padding='SAME'):
  """Create a convolution layer.
  Reference：
     Thanks to https://github.com/kratzert
  """
  input_channels = int(x.get_shape()[-1])
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights('weights', shape=[filter_height,
                                         filter_width,
                                         input_channels,
                                         num_filters],initializer=initializer，trainable=True)

    padded = tf.pad(input, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
    conv = tf.nn.conv2d(padded, weights,
        strides=[1, 1, 1, 1], padding='VALID')

    normalized = _norm(conv, is_training, norm)

    if activation == 'relu':
      output = tf.nn.relu(normalized)
    if activation == 'tanh':
      output = tf.nn.tanh(normalized)
    return output

  # Get number of input channels
  input_channels = int(x.get_shape()[-1])
  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k,strides=[1, stride_y, stride_x, 1],padding=padding)

  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape=[filter_height,
                                                filter_width,
                                                input_channels/groups,
                                                num_filters],trainable=True)
    biases = tf.get_variable('biases', shape=[num_filters],trainable=True)
    if groups == 1:
      conv = convolve(x, weights)
      # In the cases of multiple groups, split inputs & weights and
    else:
      # Split input and weights and convolve them separately
      input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
      weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                               value=weights)
      output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
      # Concat the convolved output together again
      conv = tf.concat(axis=3, values=output_groups)
    # Add biases
    conv_result = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
    # Apply relu function
    relu = tf.nn.relu(conv_result, name=scope.name)
    return relu


### Helpers
def _weights(name, shape, initializer='xavier',trainable=True,mean=0.0, stddev=0.02):
  """ Helper to create an initialized Variable
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initialization method
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
  Returns:
    A trainable variable
  Reference：
    Thanks to https://github.com/junyanz
  """
  if initializer =='normal':
    var = tf.get_variable(name, shape,
      initializer=tf.random_normal_initializer(mean=mean, stddev=stddev, dtype=tf.float32),trainable=trainable)
  #在contrib源码中，xavier的底层也是调用variance_scaling_initializer
  elif initiallizer =='xavier':
  	var = tf.get_variable(name, shape,initializer=tf.contrib.layers.xavier_initializer(),trainable=trainable)
  elif initiallizer =='scaling':
    var = tf.get_variable(name, shape,initializer=tf.contrib.layers.variance_scaling_initializer(),trainable=trainable)
  return var

def _biases(name, shape, constant=0.0):
  """ Helper to create an initialized Bias with constant
  """
  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))

def _norm(input, is_training, norm='instance'):
  """ Use Instance Normalization or Batch Normalization or None
  """
  if norm == 'instance':
    return _instance_norm(input)
  elif norm == 'batch':
    return _batch_norm(input, is_training)
  else:
    return input

def _batch_norm(input, is_training):
  """ Batch Normalization
  """
  with tf.variable_scope("batch_norm"):
    return tf.contrib.layers.batch_norm(input,
                                        decay=0.9,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training)

def _instance_norm(input):
  """ Instance Normalization
  """
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)

### 可替换代码
def _bn(x,istrain=False):
  """BN层的另一种实现方式
  Create a batch_normalization layer
  python:
    train phase: x_normal = bn(x,true)
    test phase: x_normal = bn(x,false)
    注意事项：
    1.BN层中，缩放系数gamma和偏移系数beta是可学习参数(trainable=true)，方差sigma和均值mu则是在训练中使用batch内的统计值，
    而在测试时采用的时训练时计算出的滑动平均值。属于不需更新参数(trainable=false),使用tf.layers.batch_normalization接口时
    应添加如下代码保证每次训练及时更新BN参数:
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    2.Tensorflow中，可学习变量维护在tf.GraphKeys.TRAINABLE_VARIABLES中，因此sigma和mu并不在可学习变量表中，cheakpoint文件默认仅保存
    trainable_variables,这会导致BN层参数存储和读取错误。
    建议在保存时添加如下代码：将保存列表修改为tf.global_variables()
      saver = tf.train.Saver(var_list = tf.global_variables())
      savepath = Saver.save(sess,'here_is_your_model_path')
    建议在读取时添加如下代码：
      saver = tf.train.Saver(var_list = tf.global_variables())
      saver.restore(sess,'here_is_your_model_path')
  """
  with tf.variable_scope("batch_norm"):
    return tf.layers.batch_normalization(x,training = istrain)

def Alexnet_conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
  """Create a Alexnet convolution layer.
  Reference：
     Thanks to https://github.com/kratzert
  """

  # Get number of input channels
  input_channels = int(x.get_shape()[-1])
  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k,strides=[1, stride_y, stride_x, 1],padding=padding)

  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    weights = tf.get_variable('weights', shape=[filter_height,
                                                filter_width,
                                                input_channels/groups,
                                                num_filters],trainable=True)
    biases = tf.get_variable('biases', shape=[num_filters],trainable=True)
    if groups == 1:
      conv = convolve(x, weights)
      # In the cases of multiple groups, split inputs & weights and
    else:
      # Split input and weights and convolve them separately
      input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
      weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                               value=weights)
      output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
      # Concat the convolved output together again
      conv = tf.concat(axis=3, values=output_groups)
    # Add biases
    conv_result = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
    # Apply relu function
    relu = tf.nn.relu(conv_result, name=scope.name)
    return relu
