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

  def optimize(self):
    
  def netloss():
    loss = tf.reduce_mean((tf.))
