#!/usr/bin/python
#coding:utf-8
import os
import logging
import tensorflow as tf
from datetime import datetime
from model import Mnet10
from datareader import image_datareader

FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_string('model', '/home/dell/Tensorflow/test/checkpoints/20190113-2155', 'model path (.pb) or (.ckpt)')
tf.flags.DEFINE_string('input_image_folder', './images', 'Path of input images')

tf.flags.DEFINE_integer('image_width', 128, 'width of image, default: 128')
tf.flags.DEFINE_integer('image_height', 128, 'height of image, default: 128')
tf.flags.DEFINE_integer('image_channals', 1, 'channal of image, default: 1')


def ckpt_prediction(model_path,classes):

  graph = tf.Graph()
  with graph.as_default():

    mnet10 = Mnet10(is_training=False, keep_prob = 1)
   
    reader = image_datareader(FLAGS.input_image_folder, image_height=FLAGS.image_height, image_width=FLAGS.image_width, 
                      image_mode='L',image_type='jpg', name='input')
    image_number,images_list,image_batch = reader.read_image_batch()

    
    #获取最后一个全连接层fc2的输出
    prediction_fc = mnet10.model(image_batch=image_batch)
    saver = tf.train.Saver()
    
  with tf.Session(graph=graph) as sess:
    #checkpoint = tf.train.get_checkpoint_state(FLAGS.model)
    meta_graph_path = model_path + ".meta"
    restore = tf.train.import_meta_graph(meta_graph_path)
    restore.restore(sess, model_path)

    #sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #设置logging同时输出到控制台和log文件
    DATE_FORMAT = "%m%d%Y %H:%M:%S"
    LOG_FORMAT = "%(asctime)s - %(levelname)s : -%(message)s"
    formatter = logging.Formatter(LOG_FORMAT,datefmt = DATE_FORMAT)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('prediction.log',mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    try:
      prediction_val = sess.run(prediction_fc)
      print(prediction_val)
      for i in range(image_number):
        logging.info('-----------The image: %d:-------------' % i)
        logging.info('  picture name   : {}'.format(images_list[i]))
        logging.info('  prediction_label   : {}'.format(prediction_val[i]))
        logging.info('  prediction_class   : {}'.format(classes[prediction_val[i]]))
    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      coord.join(threads)

def pb_prediction():
  return

def prediction():
  classes = ['2S1','BMP2','BRDM_2','BTR70','BTR_60','D7','T62','T72','ZIL131','ZSU_23_4']
  checkpoint = tf.train.get_checkpoint_state(FLAGS.model)
  model_path = checkpoint.model_checkpoint_path
  print('预测模型名：')
  print(model_path)
  ckpt_prediction(model_path,classes)

def main(unused_argv):
  prediction()

if __name__ == '__main__':
  tf.app.run()
