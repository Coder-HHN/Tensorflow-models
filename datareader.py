#!/usr/bin/python
#coding:utf-8
import os
import sys
import tensorflow as tf
import numpy as np 

from PIL import Image as Img
import utils

"""
模块说明：
    数据读取类
功能说明：
    1.从TFrecord文件中读取样例，组成批样例/单个样例输入到网络中
    2.依照选择进行数据扩充操作（该操作应在生成TFrecord文件时选择）
函数说明：
    1）初始化函数
      def __init__(self): 
    2）读取TFrecords文件，实现了流水线读取中的文件名队列
      def read_and_decode(tfrecord_files,image_height,image_width,image_mode,is_shuffle=True):
    3）流水线输出单个样例或批样例，对2）进行了进一步封装，实现了样例队列
      def pipeline_read(dataset_type,tfrecord_path,image_height,image_width,image_mode,is_batch=False,is_shuffle=True,
    batch_size=32,min_queue_examples=1000,num_threads=8):
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
      生成的TFrecord文件将会按照2中介绍的默认样式存储。
    2.在读取TFrecord文件时，默认TFrecoed文件放在TFrecord文件夹下，若TFrecord文件为单文件，则训练集为train.tfrecord,测试集为test.tfrecord
      若TFrecord文件为多个文件，则训练集的tfrecord文件均在train文件夹下,测试集的tfrecord文件均在test文件夹下。
      单文件：                             多文件：
      TFrecord                               TFrecord 
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

class tf_datareader:

  def __init__(self, tfrecord_path, image_height=128, image_width=128, image_mode='L', 
  	batch_size=64, min_queue_examples=1024, num_threads=8, name=''):
    """初始化函数
    Args: 
      tfrecord_path: string, TFrecord文件夹路径(eg: ./TFrecord)
      image_height: int, 图像高度
      image_width: int, 图像宽度
      image_mode: string, 图像模式
      batch_size: int, 批样例大小
      min_queue_examples：int, 预读取min_queue_examples个样例进入队列，然后随机抽取batch_size个组成batch,该值越大样例随机性越高，占用内存越大
      num_threads: int，线程数
    """
    self.tfrecord_path = tfrecord_path
    self.image_height = image_height
    self.image_width = image_width
    self.image_mode = image_mode
    self.batch_size = batch_size
    self.min_queue_examples = min_queue_examples
    self.num_threads = num_threads
    self.name = name

  def _preprocess(self, image):
    """ 读取并对TFrecords文件解码
        若需要处理非L/RGB/RGBA类型的图像，请自行添加代码
        python Image 读入的图像按照[height,weight,depth]维度排列
    Return:
      image: 3D tensor [image_width, image_height, image_depth]
    """
    if self.image_mode == 'L':
      image = tf.reshape(image,[self.image_width,self.image_height,1])
      image = utils.convert2float(image)
    elif self.image_mode == 'RGB':
      image = tf.reshape(image,[self.image_width,self.image_height,3])
      image = utils.convert2float(image)
    elif self.image_mode == 'RGBA':
      image = tf.reshape(image,[self.image_width,self.image_height,4])
      image = utils.convert2float(image)
      #image = tf.cast(image,tf.float32)*(1./255)-0.5
    else:
      print('The image mode must be L/RGB/RGBA!')
      sys.exit()
    return image

  def read_and_decode(self,tfrecord_files):
    """ 读取并对TFrecords文件解码
    Args:
      tfrecord_files: list, 一个包含所有TFrecords文件的list
    Return:
      [image，label]: 图像和标签
      image: 3D tensor [image_width, image_height, image_depth]
      label: int64
    """
    with tf.name_scope(self.name):
      filename_queue = tf.train.string_input_producer(tfrecord_files,shuffle=True)	#构造文件名队列 
      reader = tf.TFRecordReader()
      _,serialized_example = reader.read(filename_queue)	#返回的文件名和文件
      features = tf.parse_single_example(
        serialized_example,
          features ={
            'height':tf.FixedLenFeature([],tf.int64),
            'widith':tf.FixedLenFeature([],tf.int64),
            'label': tf.FixedLenFeature([],tf.int64),
            'image_raw':tf.FixedLenFeature([],tf.string),
          }
      )
    #将字符串转为uint8张量,这里仅读取image_raw和label，有需要读取图像尺寸信息的请自行修改
      image = tf.decode_raw(features['image_raw'],tf.uint8)	
      label = tf.cast(features['label'],tf.int64)
      image = self._preprocess(image)
      return image,label

  def pipeline_read(self,dataset_type):
    """ 流水线读取TFrecord文件
    Args:
      dataset_type: string,  该值设为train表示训练集，test表示测试集
    Returns:
      image_batch: 4D tensor [batch_size, image_width, image_height, image_depth]
      label_batch: 2D tensor [batch_size, label] 
    """
    #获取TFrecord文件夹路径下所有TFrecord文件名
    with tf.name_scope(self.name):
      TFreocrd_file_list = []
      for root,dirs,files in os.walk(self.tfrecord_path+'/'+dataset_type+'/'):
        for filetmp in files:
          if os.path.splitext(filetmp)[1]=='.tfrecord':
            TFreocrd_file_list.append(os.path.join(root,filetmp))
      #输出批样例，这里实现了样例队列
      image,label=self.read_and_decode(TFreocrd_file_list)      
      image_batch,label_batch = tf.train.shuffle_batch([image,label],batch_size = self.batch_size,
          capacity = self.min_queue_examples+3*self.batch_size,min_after_dequeue = self.min_queue_examples)
      return image_batch,label_batch

###
# 一种直接读取图片组成batch的方式，无需转为tfrecord
# 注意：这个类设计之初仅是为了prediction时读取图像数据所用，因此仅会读取图像信息，而不会读取label信息，
#      若需要将其用于training时读取图像数据，请自行修改添加代码
#     
class image_datareader():

  def __init__(self, images_folder_path, image_height=128, image_width=128, image_mode='L',image_type='jpg', name=''):
    """初始化函数
    Args: 
      images_folder_path: string, image图像文件夹路径,(eg: ./images)
      image_height: int, 图像高度
      image_width: int, 图像宽度
      image_mode: string, 图像模式
      image_type: string, 图像文件类型(eg:jpg or png)

    """
    self.images_folder_path = images_folder_path
    self.image_height = image_height
    self.image_width = image_width
    self.image_mode = image_mode
    self.image_type = image_type
    self.name = name

  def read_image_batch(self):
    """
      return: a tensor [batch, image_height, image_width, deepth]
    """
    with tf.name_scope(self.name):
      images_list = []
      for root,dirs,files in os.walk(self.images_folder_path):
        for filetmp in files:
          if os.path.splitext(filetmp)[1]=='.'+self.image_type:
            images_list.append(os.path.join(root,filetmp))

      image_number = len(images_list)
      #读取image并转为array，此时，array为[image_height, image_width, deepth]
      #对灰度图而言，array为[image_height, image_width]，需做后续处理
      image = Img.open(images_list[0])
      image = image.resize((self.image_height,self.image_width))
      image_array = np.array(image)
      
      #对arry做升维处理，变为[batch, image_height, image_width, deepth]或[batch, image_height, image_width]
      image_batch = np.expand_dims(image_array,axis=0)
      if image_number >1:
        for i in range(1,image_number):
          image = Img.open(images_list[i])
          image_array = np.array(image)
          image_array = np.expand_dims(image_array,axis=0)
          #在axis=0维度上合并图片,最终获得一个batch
          image_batch = np.append(image_batch,image_array,axis=0)
      
      #将array转为tensor
      image_batch = tf.convert_to_tensor(image_batch)
      #对灰度图而言，还需从[batch, image_height, image_width]转为[batch, image_height, image_width, 1]
      if self.image_mode =='L':
        image_batch = tf.reshape(image_batch,[-1,self.image_width,self.image_height,1])
      image_batch = utils.batch_convert2float(image_batch)
      return image_number,files,image_batch
###
def check_tf_reader():
  """ 检验reader读取的图像结果是否正确
  """

  TFrecordPath = './TFrecord'	#设置TFrecord文件夹路径
  #image_height,image_width,image_mode= utils.get_image_info('./Data/train/airplane/airplane5.png')
  image_width = 128
  image_heigth = 128
  image_mode = 'L'
  if not os.path.exists('./image'):
    os.makedirs('./image') 
  with tf.Graph().as_default():
    reader = tf_datareader('./TFrecord',image_heigth, image_width, image_mode, batch_size=1, 
    	            min_queue_examples=1024, num_threads=1024, name='datareader')
    image,label = reader.pipeline_read('train')
    #tensorflow中要求灰度图为[h,w,1]，但fromarray要求灰度图为[h,w]，因此需要处理一下
    if image_mode=='L':
      image = utils.batch_gray_reshape(image,image_heigth,image_width)
    #float存储图像显示有问题
    image = utils.batch_convert2int(image)
    with tf.Session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord = coord)
      try:
      	#执行3次，每次取13张样例，确认批样例随机读取代码的正确性
        for k in range(1):
          if not coord.should_stop():
            example,l = sess.run([image,label])
            print(example[0])
            for i in range(1):
              img = Img.fromarray(example[i],image_mode)
              img.save('image/'+str(k)+'_'+str(i)+'_Label_'+str(l[i])+'.jpg')
      except KeyboardInterrupt:
        print('Interrupted')
        coord.request_stop()
      except tf.errors.OutOfRangeError:
        print('OutOfRangeError')
        coord.request_stop()
      finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)

def check_image_reader():
  """ 检验reader读取的图像结果是否正确
  """
  imagePath = './images'
  #image_height,image_width,image_mode= utils.get_image_info('./Data/train/airplane/airplane5.png')
  image_width = 128
  image_heigth = 128
  image_mode = 'L'
  if not os.path.exists('./image'):
    os.makedirs('./image') 
  with tf.Graph().as_default():
    reader = image_datareader(imagePath,image_height=128,image_width=128,image_mode='L',image_type='jpg', name='')
    image_number,images_list,image = reader.read_image_batch()
    #tensorflow中要求灰度图为[h,w,1]，但fromarray要求灰度图为[h,w]，因此需要处理一下
    if image_mode=='L':
      image = utils.batch_gray_reshape(image,image_heigth,image_width)
    #float存储图像显示有问题
    image = utils.batch_convert2int(image)
    with tf.Session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord = coord)
      try:
        #执行3次，每次取13张样例，确认批样例随机读取代码的正确性
        for k in range(1):
          if not coord.should_stop():
            example = sess.run(image)
            print(example[0])
            for i in range(13):
              img = Img.fromarray(example[i],image_mode)
              img.save('image/'+str(k)+'_'+str(i)+'_'+images_list[i]+'.jpg')
      except KeyboardInterrupt:
        print('Interrupted')
        coord.request_stop()
      except tf.errors.OutOfRangeError:
        print('OutOfRangeError')
        coord.request_stop()
      finally:
        # When done, ask the threads to stop.
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
  check_image_reader()
