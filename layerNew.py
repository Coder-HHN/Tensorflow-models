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
    1）卷积层函数，实现了Convolution-BatchNorm-ReLU卷积模块，在tf.nn API的基础上实现了填充卷积。
      padding详解：请参照官网tf.pad()或
      def conv_block(...): 
    2）全连接层函数
      def fc(input, num_out, initializer='xavier', relu=True, name=''):
    3）最大池化层函数
      def max_pool(input, filter_height, filter_width, stride_y, stride_x, padding='SAME', name=''):
    4）dropout层函数
      def dropout(input, keep_prob, name=''):
    5）正则化函数，实现了instance normalization,batch_normal和local response normalization
      def normlayer(input, is_training, norm='instance', radius=2, alpha=2e-05, beta=0.75, bias=1.0):  	
    ### Helpers
    1）创建每一层的权重weights
      def _weights(name, shape, initializer='normal',mean=0.0, stddev=0.02)
      注意：出于实际工作需要，另外追加了xavier初始化方式，如果需要实现其它参数初始化方式请在该函数中自行添加
    2）创建每一层的偏置biases
      def _biases(name, shape, constant=0.0):

Reference：
    Thanks to https://github.com/kratzert
              https://github.com/vanhuyz/CycleGAN-TensorFlow 
"""

### Layers
def conv_block(input, filter_height, filter_width, num_filters, stride_y, stride_x, is_training=True, norm='batch', activation='relu',
         initializer='xavier', padding=None, reuse=False, name=''):
  """ A Convolution-BatchNorm-ReLU layer
  Args:
    input: 4D tensor [batch_size, image_width, image_height, channels]
    filter_height:int, height of filter
    filter_width:int, width of filter
    num_filters: int, number of filters
    stride_y：int, stride height
    stride_x：int, stride width
    is_training: boolean or BoolTensor
    norm: 'instance' or 'batch' or 'lrn' or None
    activation: 'relu' or 'tanh'
    initializer: initialization of parameters 'normal' or 'xavier' or 'scaling'
    padding：paddings is an integer tensor with shape [n, 2], where n is the rank of tensor
    reuse: boolean 
    name: string, e.g. 'conv_block_1'
  Returns:
    4D tensor:[batch, height, width, channels]
  """
   # Get number of input channels
  input_channels = int(input.get_shape()[-1])

  with tf.variable_scope(name, reuse=reuse):
  	# Create tf variables for the weights and biases of the conv layer
  	# filter: 4D tensor: [filter_height, filter_width, in_channels, out_channels]
    weights = _weights('weights', shape=[filter_height,
                                         filter_width,
                                         input_channels,
                                         num_filters],initializer=initializer,trainable=True)
    biases = _biases('biases', shape=[num_filters],trainable=True)

    #卷积前填充，默认为CONSTANT全0填充
    if not padding == None:
      padded = tf.pad(input, padding, 'CONSTANT'，constant_values=0)
    else:
      padded = input
    
    conv = tf.nn.conv2d(padded, weights,strides=[1, stride_y, stride_x, 1], padding='SAME')
    # Add biases
    conv_result = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    normalized = normlayer(conv_result, is_training, norm)

    if activation == 'relu':
      output = tf.nn.relu(normalized)
    if activation == 'tanh':
      output = tf.nn.tanh(normalized)
    return output

def fc(input, num_out, initializer='xavier', relu=True, name=''):
  """Create a fully connected-Relu layer.
  Args:
    input: 4D tensor [batch_size, image_width, image_height, channels]
    num_out:int, output channels
    initializer: initialization of parameters 'normal' or 'xavier' or 'scaling'
    relu: boolean
    name: string, e.g. 'fc_1' 
  Returns:
    2D tensor:[batch, num_out]
  """
  with tf.variable_scope(name):
    shape = input.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
      dim *= d
      flattened = tf.reshape(input,[-1,dim])
      # Create tf variables for the weights and biases
      weights = _weights('weights', shape=[dim, num_out],initializer=initializer,trainable=True)
      biases = _biases('biases', shape=[num_out],trainable=True)
      # Matrix multiply weights and inputs and add bias
      result = tf.nn.xw_plus_b(flattened, weights, biases)
    if relu:
      # Apply ReLu non linearity
      relu = tf.nn.relu(result)
      return relu
    else:
      return result

def max_pool(input, filter_height, filter_width, stride_y, stride_x, padding='SAME', name=''):
  """Create a max pooling layer.
  Args:
    input: 4D tensor [batch_size, image_width, image_height, channels]
    filter_height:int, height of filter
    filter_width:int, width of filter
    num_filters: int, number of filters
    stride_y：int, stride height
    stride_x：int, stride width
    padding：'SAME' or 'VALID'
    name: string, e.g. 'max_pool_1'
  Returns:
    4D tensor:[batch, height, width, channels]
  """
  with tf.variable_scope(name):
    return tf.nn.max_pool(input, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding)

def dropout(input, keep_prob, name=''):
  """Create a dropout layer.
  Args:
    keep_prob: The probability that each element is kept
  """
  with tf.variable_scope(name):
    return tf.nn.dropout(input, keep_prob)

def normlayer(input, is_training, norm='instance', radius=2, alpha=2e-05, beta=0.75, bias=1.0):
  """ Use Instance Normalization or Batch Normalization or None
  """
  if norm == 'instance':
    return _instance_norm(input)
  elif norm == 'batch':
    return _batch_norm(input, is_training)
  elif norm == 'lrn':
  	return _lr_normal(input,depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
  else:
    return input


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

def _biases(name, shape, trainable=True,constant=0.0):
  """ Helper to create an initialized Bias with constant
  """
  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant),trainable=trainable)

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

def _lr_normal(input, radius=2, alpha=2e-05, beta=0.75, bias=1.0):
  """Create a local response normalization layer."""
  with tf.variable_scope("local_response_norm") 
    return tf.nn.local_response_normalization(input, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias)

def safe_log(input, eps=1e-12):
  return tf.log(x + eps)

### 可替换代码(已废弃，仅做备用)
def _bn(input,istrain=False):
  """BN层的另一种实现方式
  Create a batch_normalization layer
  python:
    train phase: x_normal = bn(input,true)
    test phase: x_normal = bn(input,false)
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
    return tf.layers.batch_normalization(input,training = istrain)

def Alexnet_conv(input, filter_height, filter_width, num_filters, stride_y, stride_x, name,
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
      conv = convolve(input, weights)
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
