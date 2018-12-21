#!/usr/bin/python
#coding:utf-8
import os
import sys
import tensorflow as tf
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

class datareader:

  def __init(self,tfrecord_path,image_height=128,image_width=128,image_mode='L',is_batch=False,is_shuffle=True,
    batch_size=64,min_queue_examples=1024,num_threads=8,name=''):
    """初始化函数
    Args: 
      tfrecord_path: string, TFrecord文件夹路径
      image_height: int, 图像高度
      image_width: int, 图像宽度
      image_mode: string, 图像模式
      is_batch: bool, 单个样例输出或批样例输出
      is_shuffle: string, 是否随机打乱样例队列中样例顺序
      batch_size: int, 仅当is_batch = True时有效，批样例大小
      num_threads: int，线程数
      min_after_dequeue=：int,  预读取min_after_dequeue个样例进入队列，然后随机抽取batch_size个组成batch,该值越大样例随机性越高，占用内存越大
    """
    self.tfrecord_path = tfrecord_path
    self.image_height = image_height
    self.image_width = image_width
    self.image_mode = image_mode
    self.is_batch = is_batch
    self.is_shuffle = is_shuffle
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.min_after_dequeue = min_after_dequeue
    self.name = name

  def _preprocess(self, image):
  	#若需要处理非L/RGB/RGBA类型的图像，请自行添加代码
    if self.image_mode == 'L':
      image = tf.reshape(image,[self.image_height,self.image_width])
    elif self.image_mode == 'RGB':
      image = tf.reshape(image,[self.image_height,self.image_width,3])
      image = utils.convert2float(image)
    elif self.image_mode == 'RGBA':
      image = tf.reshape(image,[self.image_height,self.image_width,4])
      image = utils.convert2float(image)
      #image = tf.cast(image,tf.float32)*(1./255)-0.5
    else:
      print('The image mode must be L/RGB/RGBA!')
      sys.exit()
    return image

  def read_and_decode(self,tfrecord_files):
    """ 读取并对TFrecords文件解码
    Return:
      [image，label]: 图像和标签
    """
    with tf.name_scope(self.name):
      filename_queue = tf.train.string_input_producer(tfrecord_files,shuffle=self.is_shuffle)	#构造文件名队列 
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
      label = tf.cast(features['label'],tf.int32)
      image = self._preprocess(image)
      return image,label

  def pipeline_read(self,dataset_type):
    """
      args:
        dataset_type: string,  该值设为train表示训练集，test表示测试集
    """
    with tf.name_scope(self.name):
      TFreocrd_file_list = []
      for root,dirs,files in os.walk(self.tfrecord_path+'/'+dataset_type+'/'):
        for filetmp in files:
          if os.path.splitext(filetmp)[1]=='.tfrecord':
            TFreocrd_file_list.append(os.path.join(root,filetmp))
      #若输出单个样例
      if is_batch == False:
        image,label=read_and_decode(TFreocrd_file_list)
        return image,label
      #输出批样例，这里实现了样例队列
      elif is_batch == True:
        image,label=read_and_decode(TFreocrd_file_list)
        image_batch,label_batch = tf.train.shuffle_batch([image,label],batch_size = self.batch_size,
          capacity = self.min_queue_examples+3*self.batch_size,min_after_dequeue = self.min_queue_examples)
        return image_batch,label_batch

###
def check_reader():
  """检验reader读取的图像结果是否正确
  """
  TFrecordPath = './TFrecord'	#设置TFrecord文件夹路径
  classes = ['airplane','automobile','ship']
  image_height,image_width,image_mode= get_image_info('./Data/train/airplane/airplane5.png')
  if not os.path.exists('./image'):
    os.makedirs('./image') 
  with tf.Graph().as_default():
    reader = datareader('./TFrecord',image_height,image_width,image_mode,is_batch=False,is_shuffle=True,
        batch_size=64,min_queue_examples=1024,num_threads=8,name='')
    image,label = reader.pipline_read('train')
    with tf.Session() as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord = coord)
      try:
        for k in range(3):
          if not coord.should_stop():
            example,l = sess.run([image,label])
            for i in range(13):
              image = Image.fromarray(example[i],mode)
              image.save('image/'+str(k)+'_'+str(i)+'_Label_'+str(l[i])+'.jpg')
              #print(example,l)
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
  check_reader()
