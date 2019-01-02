#!/usr/bin/python
#coding:utf-8
import os
import sys
import tensorflow as tf
import layer
import utils
from datareader import datareader

class Mnet10:
  def __init__(self, input_train_file='', batch_size=64, image_height=128, image_width=128, 
  	       initializer='xavier', norm='batch', is_train=True, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
  	"""
    Args:
      input_train_file: string,  tfrecords file for training
      batch_size: integer, batch size
      image_height: integer, height of image
      image_width: integer, width of image
      initializer: initialization of parameters 'normal' or 'xavier' or 'scaling'
      norm: 'instance' or 'batch' or 'lrn' or None
      is_training: boolean, is Training phase
      learning_rate: float, initial learning rate for Adam
      beta1: float, momentum term of Adam
      beta2: float, momentum2 term of Adam
      epsilon: float, Adam
    """
      self.input_train_file = input_train_file
      self.batch_size = batch_size
      self.image_height = image_height
      self.image_width = image_width
      self.initializer = initializer
      self.norm = norm 
      self.is_training = is_training
      self.learning_rate = learning_rate
      self.beta1 = beta1
      self.beta2 = beta2
      self.epsilon = epsilon
  def model(self):
  	"""创建网络graph"""
    # 1st Layer: Convolution-BatchNorm-ReLU-pool layer
    self.conv1 = layer.conv_block(self.X, 11, 11, 64, 2, 2, is_training=True, norm='batch', initializer='xavier', name='conv_block1')
    self.pool1 = layer.max_pool(self.norm1, 3, 3, 2, 2, padding='SAME', name='pool1')

    # 2nd Layer: Convolution-BatchNorm-ReLU-pool layer
    self.conv2 = layer.conv_block(self.pool1, 7, 7, 96, 1, 1, is_training=True, norm='batch', initializer='xavier', name='conv_block2')
    self.pool2 = layer.max_pool(self.norm2, 3, 3, 2, 2, padding='SAME', name='pool2')

    # 3nd Layer: Convolution-BatchNorm-ReLU-pool layer
    self.conv3 = layer.conv_block(self.pool2, 5, 5, 96, 1, 1, is_training=True, norm='batch', initializer='xavier', name='conv_block3')
    self.pool3 = layer.max_pool(self.norm3, 3, 3, 1, 1, padding='SAME', name='pool3')

    # 3nd Layer: Convolution-BatchNorm-ReLU-pool layer
    self.conv4 = layer.conv_block(self.pool3, 3, 3, 96, 1, 1, is_training=True, norm='batch', initializer='xavier', name='conv_block4')
    self.pool4 = layer.max_pool(self.norm4, 3, 3, 1, 1, padding='SAME', name='pool4')

    # 5th Layer: ffully connected-BatchNorm-ReLU-> Dropout
    self.fc1 = layer.fc(self.pool4, 256, initializer='xavier', relu=True, is_training=True, norm='batch', name='fc1')
    self.dropout1 = layer.dropout(self.fc1, self.keep_prob, name='dropout1')
    
    # 6th Layer: fully connected layer
    self.fc2 = layer.fc(self.pool4, 10, initializer='xavier', relu=False, is_training=True, norm=None, name='fc2')

    return fc2

  def optimize(self, optimize_type, loss):
    if optimize_type == "Adam":
      optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon, name='Adam')
    elif optimize_type == "SGD":
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate, name='SGD')
    train_op = optimizer.minimize(loss)
    return train_op

  def netloss(logits):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batch)
    loss = tf.reduce_mean(cross_entropy)
    return loss
   