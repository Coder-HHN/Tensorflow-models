#!/usr/bin/python
#coding:utf-8
import os
import logging
import tensorflow as tf
from datetime import datetime
from model import Mnet10
from datareader import datareader

FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_string('model', '/home/dell/Tensorflow/test/checkpoints/20190113-1948/', 'model path (.pb) or (.ckpt)')
tf.flags.DEFINE_string('input_file', './TFrecord', 'Path of tfrecords Data Set Folder')

tf.flags.DEFINE_integer('image_height', 128, 'height of image, default: 128')
tf.flags.DEFINE_integer('image_width', 128, 'width of image, default: 128')
tf.flags.DEFINE_integer('batch_size', '64', 'batch size, default: 64')
tf.flags.DEFINE_integer('test_iter', '100', 'test_iter = size_of_test_data/batch_size , default: 100')

def ckpt_test(model_path):
  graph = tf.Graph()
  with graph.as_default():
    mnet10 = Mnet10()
    #设置管道读取
    input_reader = datareader(FLAGS.input_file, image_height=FLAGS.image_height, image_width=FLAGS.image_width,
         image_mode='L', batch_size=FLAGS.batch_size, min_queue_examples=1024, num_threads=8, name='Input')
    #读取训练集数据
    image_batch,label_batch = input_reader.pipeline_read('test')
    
    loss,accuracy = mnet10.model(image_batch=image_batch,label_batch=label_batch)
    saver = tf.train.Saver()
    
  with tf.Session(graph=graph) as sess:
    #checkpoint = tf.train.get_checkpoint_state(FLAGS.model)
    meta_graph_path = model_path + ".meta"
    restore = tf.train.import_meta_graph(meta_graph_path)
    restore.restore(sess, model_path)

    sess.run(tf.global_variables_initializer())
    test_iter = 0
    max_test_iter = FLAGS.test_iter
    test_accuracy_total = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
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

    try:
      for i in range(max_test_iter): 
        if not coord.should_stop():
          test_loss_val,accuracy_val = sess.run([loss,accuracy])
          #test_loss_val,accuracy_val,fc_val,label_y_val,label_origin_val = sess.run([train_op,loss,accuracy,fc,label_y,label_origin])
          if test_iter % 200 == 0:
            logging.info('-----------Batch: %d:-------------' % test_iter)
            logging.info('  test_loss   : {}'.format(test_loss_val))
            logging.info('  test_accuracy   : {}%'.format(accuracy_val*100))
            #logging.info('  fc   : {}'.format(fc_val))
            #logging.info('  label_y   : {}'.format(label_y_val))
            #logging.info('  label_origin   : {}'.format(label_origin_val))
          test_accuracy_total = test_accuracy_total+accuracy_val
          test_iter += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      logging.info('-----------Test Finish-------------')
      logging.info('  test_accuracy_average   : {}'.format(test_accuracy_total/max_test_iter))
      coord.request_stop()
      coord.join(threads)
def test():
  ckpt_test()

def main(unused_argv):
  test()

if __name__ == '__main__':
  tf.app.run()
