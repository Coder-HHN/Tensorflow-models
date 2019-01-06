import os
import logging
import tensorflow as tf
from Mnet10 import Mnet10
from datareader import Datareader

FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_string('input_file', './Data/train', 'Path of tfrecords Data Set Folder')
tf.flags.DEFINE_string('norm', 'batch', '[instance, batch, lrn, None] use instance norm or batch norm or lrn, default: batch')
tf.flags.DEFINE_string('initializer', 'xavier', 'Initialization method, normal or xavier or scaling, default: xavier')

tf.flags.DEFINE_integer('max_train_step', 30000, 'max train step, default: 30000')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size, default: 64')
tf.flags.DEFINE_integer('image_height', 128, 'height of image, default: 128')
tf.flags.DEFINE_integer('image_width', 128, 'width of image, default: 128')

tf.flags.DEFINE_bool('is_training', True, 'Training phase or test phase, default: True')

tf.flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate, default: 0.0002')
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
  	Ment10 = Ment10(
        input_file=FLAGS.input_file,
        batch_size=FLAGS.batch_size, 
        image_height=FLAGS.image_height, 
        image_width=FLAGS.image_width,
        initializer=FLAGS.initializer, 
        norm=FLAGS.norm, 
        is_train=FLAGS.is_train, 
        learning_rate=FLAGS.learning_rate, 
        beta1=FLAGS.beta1, 
        beta2=0.999, 
        epsilon=1e-08
  	)
    loss = Ment10.model()
    train_op = Ment10.optimize('Adam',loss)

    saver = tf.train.Saver()
    
  with tf.Session(graph=graph) as sess:
   """ if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0
      max_train_step = FLAGS.max_train_step"""
    sess.run(tf.global_variables_initializer())
    step = 0
    max_train_step = FLAGS.max_train_step
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
      for i in range(max_train_step):	
        if not coord.should_stop():
          [train_loss_val,accuracy] = sess.run(train_op)
          if step % 200 == 0:
            logging.info('-----------Step %d:-------------' % step)
            logging.info('  train_loss   : {}'.format(train_loss_val))
            logging.info('  train_loss   : {}'.format(accuracy))
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
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
