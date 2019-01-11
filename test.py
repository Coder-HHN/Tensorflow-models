#!/usr/bin/python
#coding:utf-8
import os
import logging
import tensorflow as tf
from datetime import datetime
from model import Mnet10
from datareader import datareader

FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_string('model', '', 'model path (.pb) or (.ckpt)')
tf.flags.DEFINE_string('input_img', None, 'input image path (.jpg)')
tf.flags.DEFINE_string('input_file', './TFrecord', 'Path of tfrecords Data Set Folder')

tf.flags.DEFINE_integer('image_height', 128, 'height of image, default: 128')
tf.flags.DEFINE_integer('image_width', 128, 'width of image, default: 128')
tf.flags.DEFINE_integer('batch_size', '64', 'batch size, default: 64')
tf.flags.DEFINE_integer('test_iter', '10000', 'test_iter = size_of_test_data/batch_size , default: 10000')

def ckpt_test(model=None,input=None,is_single_model=False,image_batch=None,label_batch=None):
  """ 对已有模型执行测试操作，返回模型测试精度，输入模型保存文件为*.ckpt
  Args:
    model: string, 模型文件路径
      注意：若要对训练中保存的所有模型进行测试，请将is_single_model设置为False，同时，此时应输入保存模型的文件夹的路径(eg:./ckpt/)
      若要指定测试某一个模型，请将is_single_model设置为True，此时应输入全路径（eg:/ckpt/your_model_name.ckpt）
      当is_single_model=True且 model=None时，默认仅测试文件夹中最新保存的模型
    input: sting, 输入数据,TFrecord文件夹路径
    is_single_model：对所有保存模型进行测试(False)，或对某一个模型进行测试(True)
  Return: 
    accuracy: float, 准确率
  """
  if is_single_model = False:
    checkpoint = tf.train.get_checkpoint_state(model)
    logging.info('-----------Test Begin -------------' )
    logging.info('Models: %s' % checkpoint)
    if checkpoint and checkpoint.all_model_checkpoint_paths:
      for model_path in checkpoint.all_model_checkpoint_paths:
        # 载入图结构，保存在.meta文件中        	
        saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path +'.meta')
          # 载入参数文件，restore会依据model_checkpoint_path自行寻找
          saver.restore(sess,model_path)
          global_step=model_path.split('/')[-1].split('-')[-1]

    else
      print('The Model Save Folder is empty!')
      sys.exit()
  elif is_single_model = True:  
    if model == None
  
    else


      checkpoint = tf.train.get_checkpoint_state(model)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(model))
      step = int(meta_graph_path.split("-")[2].split(".")[0])

def pb_test():


def test():
  if FLAGS.input_img is not None:


def main(unused_argv):
  #读入测试集数据
  input_reader = datareader(FLAGS.input_file, image_height=FLAGS.image_height, image_width=FLAGS.image_width,
         image_mode='L', batch_size=FLAGS.batch_size, min_queue_examples=1024, num_threads=8, name='Input')
  image_batch,label_batch = input_reader.pipeline_read('test')

  #设置logging同时输出到控制台和log文件
  DATE_FORMAT = "%m%d%Y %H:%M:%S"
  LOG_FORMAT = "%(asctime)s - %(levelname)s : -%(message)s"
  formatter = logging.Formatter(LOG_FORMAT,datefmt = DATE_FORMAT)

  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  file_handler = logging.FileHandler('test.log',mode='w')
  file_handler.setLevel(logging.INFO)
  file_handler.setFormatter(formatter)

  stream_handler = logging.StreamHandler()
  stream_handler.setLevel(logging.INFO)
  stream_handler.setFormatter(formatter)

  logger.addHandler(file_handler)
  logger.addHandler(stream_handler)
  test()

if __name__ == '__main__':
  tf.app.run()