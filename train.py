# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:38:23 2018

@author: shirhe-lyh
"""

import os
import tensorflow as tf

import model
import preprocessing

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('record_path', 
                    '/data2/raycloud/deep_image_matting/' +
                    'train_boundary.record',
                    'Path to training tfrecord file.')
flags.DEFINE_string('checkpoint_path', 
                    '/data2/raycloud/model_zoo/' +
                    'vgg_16.ckpt', 
                    'Path to pretrained model.')
flags.DEFINE_string('logdir', './training', 'Path to log directory.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate_decay_factor', 0.1, 
                   'Learning rate decay factor.')
flags.DEFINE_float('num_epochs_per_decay', 2.0,
                   'Number of epochs after which learning rate decays. ' +
                   'Note: this flag counts epochs per clone but aggregates ' +
                   'per sync replicas. So 1.0 means that each clone will go ' +
                   'over full epoch individually, but replicas will go once ' +
                   'across all replicas.')
flags.DEFINE_integer('num_samples', 40500, 'Number of samples.')
flags.DEFINE_integer('num_steps', 100000, 'Number of steps.')
flags.DEFINE_integer('batch_size', 4, 'Batch size')
flags.DEFINE_integer('input_size', 320, 'The input size of network')

FLAGS = flags.FLAGS


def get_record_dataset(record_path,
                       reader=None, 
                       num_samples=50000, 
                       num_classes=None):
    """Get a tensorflow record file.
    
    Args:
        
    """
    if not reader:
        reader = tf.TFRecordReader
        
    keys_to_features = {
        'image_fg/encoded': 
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': 
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image_bg/encoded': 
            tf.FixedLenFeature((), tf.string, default_value=''),
        'trimap/encoded': 
            tf.FixedLenFeature((), tf.string, default_value=''),
        'alpha_matte/encoded': 
            tf.FixedLenFeature((), tf.string, default_value=''),
        }
        
    items_to_handlers = {
        'image_fg': slim.tfexample_decoder.Image(
            image_key='image_fg/encoded', format_key='image/format'),
        'image_bg': slim.tfexample_decoder.Image(
            image_key='image_bg/encoded', format_key='image/format'),
        'trimap': slim.tfexample_decoder.Image(
            image_key='trimap/encoded', format_key='image/format', 
            channels=1),
        'alpha_matte': slim.tfexample_decoder.Image(
            image_key='alpha_matte/encoded', format_key='image/format',
            channels=1),
        }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    
    labels_to_names = None
    items_to_descriptions = {
        'image_fg': 'The foreground image.',
        'image_bg': 'The background image.',
        'trimap': 'The trimap.',
        'alpha_matte': 'The alpha matte.'}
    return slim.dataset.Dataset(
        data_sources=record_path,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        num_classes=num_classes,
        items_to_descriptions=items_to_descriptions,
        labels_to_names=labels_to_names)
    
    
def configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/
        train_image_classifier.py
    
    Args:
        num_samples_per_epoch: he number of samples in each epoch of training.
        global_step: The global_step tensor.
        
    Returns:
        A `Tensor` representing the learning rate.
    """
    decay_steps = int(num_samples_per_epoch * FLAGS.num_epochs_per_decay /
                      FLAGS.batch_size)
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
    
    
def get_init_fn(checkpoint_exclude_scopes=None):
    """Returns a function run by che chief worker to warm-start the training.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/
        train_image_classifier.py
    
    Note that the init_fn is only run when initializing the model during the 
    very first global step.
    
    Args:
        checkpoint_exclude_scopes: Comma-separated list of scopes of variables
            to exclude when restoring from a checkpoint.
    
    Returns:
        An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None
    
    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.logdir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists ' +
            'in %s' % FLAGS.logdir)
        return None
    
    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in 
                     checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)
    
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    
    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=True)


def main(_):
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    num_samples = FLAGS.num_samples
    dataset = get_record_dataset(FLAGS.record_path, num_samples=num_samples)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image_fg, image_bg, trimap, alpha = data_provider.get(
        ['image_fg', 'image_bg', 'trimap', 'alpha_matte'])

    size = FLAGS.input_size
    size_bg = int(1.0 * size)
    image_fg = preprocessing.border_expand_and_resize(image_fg, size, size)
    trimap = preprocessing.border_expand_and_resize(trimap, size, size, 1)
    alpha = preprocessing.border_expand_and_resize(alpha, size, size, 1)
    image_bg = preprocessing.random_crop(image_bg, size_bg, size_bg)
        
    images_fg, images_bg, trimaps, alpha_mattes = tf.train.batch(
        [image_fg, image_bg, trimap, alpha],
        batch_size=FLAGS.batch_size,
        #capacity=5*FLAGS.batch_size,
        allow_smaller_final_batch=True)
    
    cls_model = model.Model(is_training=True, default_image_size=size)
    preprocessed_inputs = cls_model.preprocess(trimaps, None, images_fg,
                                               images_bg, alpha_mattes)
    prediction_dict = cls_model.predict(preprocessed_inputs)
    loss_dict = cls_model.loss(prediction_dict, preprocessed_inputs)
    loss = loss_dict['loss']
    tf.summary.scalar('loss', loss)
    tf.summary.image('gt_alpha_mattes', 255 * alpha_mattes, max_outputs=5)
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    for key, value in postprocessed_dict.items():
        tf.summary.image(key, value, max_outputs=5)

#    global_step = slim.create_global_step()
#    learning_rate = configure_learning_rate(num_samples, global_step)
#    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, 
#                                           momentum=0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
    train_op = slim.learning.create_train_op(loss, optimizer,
                                             summarize_gradients=True)
#    tf.summary.scalar('learning_rate', learning_rate)
    
    init_fn = get_init_fn(checkpoint_exclude_scopes='vgg_16/fc8')
    
    slim.learning.train(train_op=train_op, logdir=FLAGS.logdir, 
                        init_fn=init_fn, number_of_steps=FLAGS.num_steps,
                        save_summaries_secs=20,
                        save_interval_secs=600)
    
if __name__ == '__main__':
    tf.app.run()

