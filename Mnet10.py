#!/usr/bin/python
#coding:utf-8
import os
import sys
import tensorflow as tf
import layer
import utils
from datareader import datareader

class MNET_10:
  def __init__(self,):
    
  def model(self):
    """创建网络graph"""
    # 1st Layer:Conv(w Relu)/BatchNormal
    self.conv1 = layer.conv(self.X, 11, 11, 64, 2, 2, padding='SAME', name='conv1')
    self.norm1 = layer.bn(self.conv1,istrain=True)
    self.pool1 = layer.max_pool(self.norm1, 3, 3, 2, 2, padding='SAME', name='pool1')

    # 2nd Layer:Conv(w Relu)/BatchNormal
    self.conv2 = layer.conv(self.pool1, 7, 7, 96, 1, 1, padding='SAME', name='conv2')
    self.norm2 = layer.bn(self.conv2,istrain=True)
    self.pool2 = layer.max_pool(self.norm2, 3, 3, 2, 2, padding='SAME', name='pool2')

    # 3nd Layer:Conv(w Relu)/BatchNormal
    self.conv3 = layer.conv(self.pool2, 5, 5, 96, 1, 1, padding='SAME', name='conv2')
    self.norm3 = layer.bn(self.conv3,istrain=True)
    self.pool3 = layer.max_pool(self.norm3, 3, 3, 1, 1, padding='SAME', name='pool2')

    # 3nd Layer:Conv(w Relu)/BatchNormal
    self.conv4 = layer.conv(self.pool3, 3, 3, 96, 1, 1, padding='SAME', name='conv2')
    self.norm4 = layer.bn(self.conv4,istrain=True)
    self.pool4 = layer.max_pool(self.norm4, 3, 3, 1, 1, padding='SAME', name='pool2')

    # 5th Layer: FC (w ReLu) -> Dropout
    self.fc1 = layer.fc(self.pool4,256, name='fc6')
    self.norm5 = layer.bn(self.fc1,istrain=True)
    self.dropout1 = layer.dropout(self.norm5, self.KEEP_PROB)

    # 6th Layer: FC (w ReLu)
    self.fc2 = layer.fc(self.dropout1,self.NUM_CLASSES, relu=False,name='fc6')
    logits = fc2
    return logits
  def optimize(self,op_type):
    def make_optimizer(loss, variables, name='Adam'):
      """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
          and a linearly decaying rate that goes to zero over the next 100k steps
      """
      global_step = tf.Variable(0, trainable=False)
      starter_learning_rate = self.learning_rate
      end_learning_rate = 0.0
      start_decay_step = 100000
      decay_steps = 100000
      beta1 = self.beta1
      learning_rate = (
          tf.where(
                  tf.greater_equal(global_step, start_decay_step),
                  tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                            decay_steps, end_learning_rate,
                                            power=1.0),
                  starter_learning_rate
          )

      )
      tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

      learning_step = (
          tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                  .minimize(loss, global_step=global_step, var_list=variables)
      )
      return learning_step
    
  def netloss():
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batch)
    loss = tf.reduce_mean(cross_entropy)
    return loss
   