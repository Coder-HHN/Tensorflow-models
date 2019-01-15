#!/usr/bin/python
#coding:utf-8

import os
import logging
import tensorflow as tf
from datetime import datetime
from model import Mnet10
from datareader import tf_datareader

FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_string('input_file', './TFrecord', 'Path of tfrecords Data Set Folder')
tf.flags.DEFINE_string('norm', 'batch', '[instance, batch, lrn, None] use instance norm or batch norm or lrn, default: batch')
tf.flags.DEFINE_string('initializer', 'xavier', 'Initialization method, normal or xavier or scaling, default: xavier')
tf.flags.DEFINE_string('image_mode', 'L', 'mode of image: L or RGB or RGBA')

tf.flags.DEFINE_integer('max_train_step', 100000, 'max train step, default: 30000')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size, default: 64')
tf.flags.DEFINE_integer('image_height', 128, 'height of image, default: 128')
tf.flags.DEFINE_integer('image_width', 128, 'width of image, default: 128')

tf.flags.DEFINE_bool('is_training', True, 'Training phase or test phase, default: True')

tf.flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate, default: 0.0002')
tf.flags.DEFINE_float('keep_prob', 0.8, 'the probability that each element is kept')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')

tf.flags.DEFINE_string('load_model', None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')


def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}".format(current_time)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass

  graph = tf.Graph()
  with graph.as_default():
    mnet10 = Mnet10(
        initializer=FLAGS.initializer, 
        norm=FLAGS.norm, 
        is_training=FLAGS.is_training, 
        learning_rate=FLAGS.learning_rate, 
        keep_prob = FLAGS.keep_prob,
        beta1=FLAGS.beta1, 
        beta2=0.999, 
        epsilon=1e-08
    )

    #设置管道读取
    input_reader = tf_datareader(FLAGS.input_file, image_height=FLAGS.image_height, image_width=FLAGS.image_width,
         image_mode=FLAGS.image_mode, batch_size=FLAGS.batch_size, min_queue_examples=1024, num_threads=8, name='Input')
    #读取训练集数据
    image_batch,label_batch = input_reader.pipeline_read('train')
    
    loss,accuracy = mnet10.model(image_batch=image_batch,label_batch=label_batch)
    #loss,accuracy,fc,label_y,label_origin = mnet10.model(image_batch=image_batch,label_batch=label_batch)
    train_op = mnet10.optimize('SGD',loss)

    saver = tf.train.Saver()
    
  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    max_train_step = FLAGS.max_train_step
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    #设置logging同时输出到控制台和log文件
    DATE_FORMAT = "%m%d%Y %H:%M:%S"
    LOG_FORMAT = "%(asctime)s - %(levelname)s : -%(message)s"
    formatter = logging.Formatter(LOG_FORMAT,datefmt = DATE_FORMAT)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('train.log',mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    try:
      for i in range(max_train_step):	
        if not coord.should_stop():
          _,train_loss_val,accuracy_val = sess.run([train_op,loss,accuracy])
          #_,train_loss_val,accuracy_val,fc_val,label_y_val,label_origin_val = sess.run([train_op,loss,accuracy,fc,label_y,label_origin])
          if step % 200 == 0:
            logging.info('-----------Step %d:-------------' % step)
            logging.info('  train_loss   : {}'.format(train_loss_val))
            logging.info('  train_accuracy   : {}%'.format(accuracy_val*100))
            #logging.info('  fc   : {}'.format(fc_val))
            #logging.info('  label_y   : {}'.format(label_y_val))
            #logging.info('  label_origin   : {}'.format(label_origin_val))
          if step % 10000 == 0:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)
          step += 1

    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  #logging.basicConfig(level=logging.INFO)
  tf.app.run()
