#!/usr/bin/python
#coding:utf-8
import os
import sys
import tensorflow as tf
from PIL import Image

"""
模块说明：
    数据处理类
功能说明：
    1.将不同类型数据转为TFrecord文件
    2.依照选择进行数据扩充操作（该操作应在生成TFrecord文件时选择）
函数说明：
      1）初始化函数
      def __init__(self): 
      2）数据转为TFrecord文件
      def write_to_TFrecord(dataset_type,dataset_path,classes,image_height,image_width,is_devide=True,by_class=True,num_per_tf=10000):

文件夹样板说明：
    1.在将数据集转为TFrecord文件时，默认test放置测试图片，train放置训练图片，train和test共同放置于Data文件夹下，
      图片已经按照类别分置于train和test下。若采用其它放置方式，请自行修改代码保证正确读取数据。
      Data      
      |
      |---------train
      |          |
      |          |------class1------数据
      |          |------class2
      |
      |---------test
      |          |
      |          |------class1
      |          |------class2
      生成的TFrecord文件将会按照下面介绍的默认样式存储。
    2.在读取TFrecord文件时，默认TFrecoed文件放在TFrecord文件夹下，若TFrecord文件为单文件，则训练集为train.tfrecord,测试集为test.tfrecord
      若TFrecord文件为多个文件，则训练集的tfrecord文件均在train文件夹下,测试集的tfrecord文件均在test文件夹下。
      单文件：                             多文件：
      Data                                  Data 
      |                                      |
      |---------train                        |---------train
      |          |                           |          |------1.tfrecord
      |          |------train.tfrecord       |          |------2.tfrecord               
      |                                      |
      |---------test                         |---------test
      |          |                           |          |
      |          |------test.tfrecord        |          |------1.tfrecord
                                             |          |------2.tfrecord     
"""

class dataprocess():

  #初始化函数
  def __init__(self,dataset_path,classes,image_height=128,image_width=128,is_devide=True,by_class=True,num_per_tf=1000):
    """
      Args: 
        dataset_path: string,  原始数据集存储路径,即Data文件夹所在路径
        classes：list, 要设置的标签列表，例：[car,airplane,ship,...],建议采用方括号
        image_height: integer, 图像高度
        image_width: integer, 图像宽度
        is_devide: bool, 是否对数据集进行分割来生成多个tfrecord文件
        by_class: bool, 按照类别or样例数量来分割数据集
        num_per_tf: integer, 仅当by_class=False时有效，每个tfrecord文件存储的样例个数
    """
    self.dataset_path = dataset_path
    self.classes = classes
    self.image_height = image_height
    self.image_width = image_width
    self.is_devide = is_devide
    self.by_class = by_class
    self.num_per_tf = num_per_tf
    self.name = name

  #TFrecord支持的三种数据类型
  def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

  def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

  def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

  #图像编码函数
  def encode_img(self,img_path,label):
    """
      图像编码函数,目的是为了将write_to_TFrecord函数中不同逻辑当中的重复代码进行简化
      若需要修改图像封装格式，请在此处修改即可。
      args:
        img_path: string, 单张图片
        label: int, 图像标签，从0开始
      return：
        一个编码后的example
    """
    img = Image.open(img_path)
    img = img.resize((self.image_height,self.image_width))
    img_raw = img.tobytes()    #图片转化为二进制格式
    example = tf.train.Example(    #对label和img数据进行封装
      features = tf.train.Features(
        feature = {
          'height':_int64_feature(img.size[0]),
          'widith':_int64_feature(img.size[1]),
          'label': _int64_feature(label),
          'img_raw':_bytes_feature(img_raw)
        }
      )
    )
    return example

  #读入图像和标签并转为TFrecords文件
  def write_to_TFrecord(self,dataset_type):
    """
      Args: 
        dataset_type: string,  该值设为train表示正在转换训练集，test表示正在转换测试集
    """
    if not os.path.exists('TFrecord/'+dataset_type+'/'):
      os.makedirs('TFrecord/'+dataset_type+'/')
    if is_devide == True:
      #生成的TF文件默认存储在TFrecord文件夹下
      Destfile = tf.python_io.TFRecordWriter('TFrecord/'+dataset_type+'/'+dataset_type+'.tfrecord')
      for index,name in enumerate(self.classes):
        class_path = self.dataset_path+'/'+dataset_type+'/'+name
          if not os.path.exists(class_path):
            print(class_path+'is not exist!')
            sys.exit()
          for img_name in os.listdir(class_path):
            img_path = class_path+'/'+img_name   #每张图片的地址
            example = encode_img(img_path,index)    #编码图像
            Destfile.write(example.SerializeToString())	 #序列化为字符串
      Destfile.close()
    elif is_devide = False:
      if by_class == True:
        for index,name in enumerate(self.classes):
          class_path = self.dataset_path+'/'+dataset_type+'/'+name
          if not os.path.exists(class_path):
            print(class_path+'is not exist!')
            sys.exit()
          #生成的TF文件按train和test分类别存放
          Destfile = tf.python_io.TFRecordWriter('TFrecord/'+dataset_type+'/'+name+'.tfrecord')
          for img_name in os.listdir(class_path):
            img_path = class_path+'/'+img_name   #每张图片的地址
            example = encode_img(img_path,index)    #编码图像
            Destfile.write(example.SerializeToString())	 #序列化为字符串
          Destfile.close()
      elif by_class == False:
        icount_start = 0
        icount_end = icount_start+devideNum
        #生成的TF文件按train和test分类别存放
        Destfile = tf.python_io.TFRecordWriter('TFrecord/'+dataset_type+'/'+dataset_type+'_'+str(icount_start)+
          '_'+str(icount_end)+'.tfrecord')
        for index,name in enumerate(self.classes):
          class_path = self.dataset_path+'/'+dataset_type+'/'+name
          if not os.path.exists(class_path):
            print(class_path+'is not exist!')
            sys.exit()
          length = len(os.listdir(class_path))
          for img_name in os.listdir(class_path):
            img_path = class_path+'/'+img_name    #每张图片的地址
            example = encode_img(img_path,index)    #编码图像
            Destfile.write(example.SerializeToString())    #序列化为字符串
            length = length-1
            icount_start += 1
            if length == 0 and index == len(self.classes)-1: 
              Destfile.close()
            else:
              if icount_start%self.num_per_tf == 0:
                icount_end = icount_start+self.num_per_tf
                Destfile.close()
                Destfile = tf.python_io.TFRecordWriter('TFrecord/'+dataset_type+'/'+dataset_type+'_'+str(icount_start)+
                  '_'+str(icount_end)+'.tfrecord')

  def test_writer():
    """
      示例程序:将图像及标签写入TFrecord文件
    """
    datapath = './Data'  #设置TFrecord文件夹路径
    classes = ['airplane','automobile','ship']
    image_width = 128
    image_height = 128
    writer = dataprocess(datapath,classes,image_height,image_width)
    writer.write_to_TFrecord('train')
    writer.write_to_TFrecord('test')
if __name__ == '__main__':
  test_writer()

