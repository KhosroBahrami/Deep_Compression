
# Config 

import tensorflow as tf


tf.app.flags.DEFINE_string('model_name', 'vgg', 'model name')

tf.app.flags.DEFINE_string('histograms_dir', './histograms_', 'histogram directory')
tf.app.flags.DEFINE_string('weights_dir', './weights_', 'weigths directory')


tf.app.flags.DEFINE_integer('all_steps', 1500, 'number of steps')
tf.app.flags.DEFINE_integer('full_training_steps', 500, 'number of steps for full training')
tf.app.flags.DEFINE_integer('pruning_fine_tunning_steps', 1000, 'steps for pruning & fine tunning')

tf.app.flags.DEFINE_integer('pruning_fine_tunning_step', 500, 'step for pruning & fine tunning')
tf.app.flags.DEFINE_integer('quantization_fine_tunning_step', 1000, 'step for quantization & fine tunning')






FLAGS = tf.app.flags.FLAGS


