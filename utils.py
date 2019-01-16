#!/usr/bin/python
#coding:utf-8

import tensorflow as tf
from PIL import Image as Img

def get_image_info(image_path):
  """ 获取图像信息，为read_and_decode函数提供参数，若已知图像格式则不需调用
    PIL中图像mode有9种，1/L/P/RBG/RBGA/CMYK/YCbCr/I/F
    在此处仅介绍常用的三种，其余请自行查询
    L：8位像素灰度图，1通道
    RGB：3x8位像素彩色图，3通道
    RGBA：4x8位像素彩色图+透明通道，4通道
  Args:
    image_path: string, 选定图像的路径
  Return: 
    image_height: int, 图像高度
    image_width: int, 图像宽度
    image_mode: string, 图像模式   
  """
  image = Img.open(image_path)
  image_height = image.size[0]
  image_width = image.size[1]
  image_mode = image.mode
  return  image_height,image_width,image_mode

def convert2int(image):
  """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
  """
  image = image*255
  image = tf.cast(image,tf.uint8)
  return image
def convert2float(image):
  """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
  """
  return tf.cast(image,tf.float32)*(1./255)

def gray_reshape(images, image_heigth, image_width):
  return tf.reshape(image,[image_width,image_heigth])

def batch_convert2int(images):
  """
  Args:
    images: 4D float tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D int tensor
  """
  return tf.map_fn(convert2int, images, dtype=tf.uint8)

def batch_convert2float(images):
  """
  Args:
    images: 4D int tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D float tensor
  """
  return tf.map_fn(convert2float, images, dtype=tf.float32)

def batch_gray_reshape(images, image_heigth, image_width):
  return tf.map_fn(lambda x: gray_reshape(x,image_width,image_heigth), images, dtype=tf.float32)
