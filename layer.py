""" 
    对神经网络中各类型层的封装
    @author: HHN
"""

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):

  """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
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

def fc(x, num_out, name, relu=True):
  """Create a fully connected layer."""
  with tf.variable_scope(name) as scope:
    shape = x.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
      dim *= d
      flattened = tf.reshape(x,[-1,dim])
      # Create tf variables for the weights and biases
      weights = tf.get_variable('weights', shape=[dim, num_out],
                                trainable=True)
      biases = tf.get_variable('biases', [num_out], trainable=True)
      # Matrix multiply weights and inputs and add bias
      result = tf.nn.xw_plus_b(flattened, weights, biases, name=scope.name)
    if relu:
      # Apply ReLu non linearity
      relu = tf.nn.relu(result)
      return relu
    else:
      return result
    
def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
  """Create a max pooling layer."""
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides=[1, stride_y, stride_x, 1],
                        padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
  """Create a local response normalization layer."""
  return tf.nn.local_response_normalization(x, depth_radius=radius,
                                            alpha=alpha, beta=beta,
                                            bias=bias, name=name)

def dropout(x, keep_prob):
  """Create a dropout layer."""
  return tf.nn.dropout(x, keep_prob)

def bn(x,istrain=False):
  """Create a batch_normalization layer
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
  return tf.layers.batch_normalization(x,training = istrain)
