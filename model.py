# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:11:59 2018

@author: shirhe-lyh
"""

import tensorflow as tf

from tensorflow.contrib.slim import nets

import preprocessing

slim = tf.contrib.slim
    
        
class Model(object):
    """xxx definition."""
    
    def __init__(self, is_training,
                 default_image_size=320,
                 first_stage_alpha_loss_weight=1.0,
                 first_stage_image_loss_weight=1.0,
                 second_stage_alpha_loss_weight=1.0):
        """Constructor.
        
        Args:
            is_training: A boolean indicating whether the training version of
                computation graph should be constructed.
        """
        self._is_training = is_training
        self._default_image_size = default_image_size
        self._first_stage_alpha_loss_weight = first_stage_alpha_loss_weight
        self._first_stage_image_loss_weight = first_stage_image_loss_weight
        self._second_stage_alpha_loss_weight = second_stage_alpha_loss_weight
        
    def preprocess(self, trimaps, images=None, images_forground=None, 
                   images_background=None, alpha_mattes=None):
        """preprocessing.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            trimaps: A float32 tensor with shape [batch_size,
                height, width, 1] representing a batch of trimaps.
            images: A float32 tensor with shape [batch_size, height, width,
                3] representing a batch of images. Only passed values in case
                of test (i.e., in training case images=None).
            images_foreground: A float32 tensor with shape [batch_size,
                height, width, 3] representing a batch of foreground images.
            images_background: A float32 tensor with shape [batch_size,
                height, width, 3] representing a batch of background images.
            alpha_mattes: A float32 tensor with shape [batch_size,
                height, width, 1] representing a batch of groundtruth masks.
            
            
        Returns:
            The preprocessed tensors.
        """
        def _random_crop(t):
            num_channels = t.get_shape().as_list()[2]
            return preprocessing.random_crop_background(
                t, output_height=self._default_image_size,
                output_width=self._default_image_size, 
                channels=num_channels)
        
        def _border_expand_and_resize(t):
            return preprocessing.border_expand_and_resize(
                t, output_height=self._default_image_size,
                output_width=self._default_image_size)
            
        def _border_expand_and_resize_g(t):
            return preprocessing.border_expand_and_resize(
                t, output_height=self._default_image_size,
                output_width=self._default_image_size,
                channels=1)
            
        preprocessed_images_fg = None
        preprocessed_images_bg = None
        preprocessed_alpha_mattes = None
        preprocessed_trimaps = tf.map_fn(_border_expand_and_resize_g, trimaps)
        preprocessed_trimaps = tf.to_float(preprocessed_trimaps)
        if self._is_training:
            preprocessed_images_fg = tf.map_fn(_border_expand_and_resize, 
                                               images_forground)
            preprocessed_alpha_mattes = tf.map_fn(_border_expand_and_resize_g, 
                                                  alpha_mattes)
            images_background = tf.to_float(images_background)
            preprocessed_images_bg = tf.map_fn(_random_crop, images_background)
        
            preprocessed_images_fg = tf.to_float(preprocessed_images_fg)
            preprocessed_alpha_mattes = tf.to_float(preprocessed_alpha_mattes)
            preprocessed_images = (tf.multiply(
                    preprocessed_alpha_mattes, preprocessed_images_fg) + 
                tf.multiply(
                    1 - preprocessed_alpha_mattes, preprocessed_images_bg))
        else:
            preprocessed_images = tf.map_fn(_border_expand_and_resize, images)
            preprocessed_images = tf.to_float(preprocessed_images)
            
        preprocessed_dict = {'images_fg': preprocessed_images_fg,
                             'images_bg': preprocessed_images_bg,
                             'alpha_mattes': preprocessed_alpha_mattes,
                             'images': preprocessed_images,
                             'trimaps': preprocessed_trimaps}
        return preprocessed_dict
    
    def predict(self, preprocessed_dict):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_dict: See The preprocess function.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        # The inputs for the first stage
        preprocessed_images = preprocessed_dict.get('images')
        preprocessed_trimaps = preprocessed_dict.get('trimaps')
        
        # VGG-16
        _, endpoints = nets.vgg.vgg_16(preprocessed_images,
                                       num_classes=1,
                                       spatial_squeeze=False,
                                       is_training=self._is_training)
        # Note: The `padding` method of fc6 of VGG-16 in tf.contrib.slim is
        # `VALID`, but the expected value is `SAME`, so we must replace it.
        net_image = endpoints.get('vgg_16/pool5')
        net_image = slim.conv2d(net_image, num_outputs=4096, kernel_size=7, 
                                padding='SAME', scope='fc6_')
        
        # VGG-16 for alpha channel
        net_alpha = slim.repeat(preprocessed_trimaps, 2, slim.conv2d, 64,
                                [3, 3], scope='conv1_alpha')
        net_alpha = slim.max_pool2d(net_alpha, [2, 2], scope='pool1_alpha')
        net_alpha = slim.repeat(net_alpha, 2, slim.conv2d, 128, [3, 3],
                                scope='conv2_alpha')
        net_alpha = slim.max_pool2d(net_alpha, [2, 2], scope='pool2_alpha')
        net_alpha = slim.repeat(net_alpha, 2, slim.conv2d, 256, [3, 3],
                                scope='conv3_alpha')
        net_alpha = slim.max_pool2d(net_alpha, [2, 2], scope='pool3_alpha')
        net_alpha = slim.repeat(net_alpha, 2, slim.conv2d, 512, [3, 3],
                                scope='conv4_alpha')
        net_alpha = slim.max_pool2d(net_alpha, [2, 2], scope='pool4_alpha')
        net_alpha = slim.repeat(net_alpha, 2, slim.conv2d, 512, [3, 3],
                                scope='conv5_alpha')
        net_alpha = slim.max_pool2d(net_alpha, [2, 2], scope='pool5_alpha')
        net_alpha = slim.conv2d(net_alpha, 4096, [7, 7], padding='SAME',
                                scope='fc6_alpha')
        
        # Concate the first stage prediction
        net = tf.concat(values=[net_image, net_alpha], axis=3)
        net.set_shape([None, self._default_image_size // 32,
                       self._default_image_size // 32, 8192])
        
        # Deconvlution
        with slim.arg_scope([slim.conv2d_transpose], stride=2, kernel_size=5):
            # Deconv6
            net = slim.conv2d_transpose(net, num_outputs=512, kernel_size=1,
                                        scope='deconv6')
            # Deconv5
            net = slim.conv2d_transpose(net, num_outputs=512, scope='deconv5')
            # Deconv4
            net = slim.conv2d_transpose(net, num_outputs=256, scope='deconv4')
            # Deconv3
            net = slim.conv2d_transpose(net, num_outputs=128, scope='deconv3')
            # Deconv2
            net = slim.conv2d_transpose(net, num_outputs=64, scope='deconv2')
            # Deconv1
            net = slim.conv2d_transpose(net, num_outputs=64, stride=1, 
                                        scope='deconv1')
        
        # Predict alpha matte
        alpha_matte = slim.conv2d(net, num_outputs=1, kernel_size=[5, 5],
                                  activation_fn=tf.nn.sigmoid,
                                  scope='AlphaMatte')

        # The inputs for the second stage
        alpha_matte_scaled = tf.multiply(alpha_matte, 255.)
        refine_inputs = tf.concat(
            values=[preprocessed_images, alpha_matte_scaled], axis=3)
        refine_inputs.set_shape([None, self._default_image_size, 
                                 self._default_image_size, 4])
        
        # Refine
        net = slim.conv2d(refine_inputs, num_outputs=64, kernel_size=[3, 3],
                          scope='refine_conv1')
        net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3],
                          scope='refine_conv2')
        net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3],
                          scope='refine_conv3')
        refined_alpha_matte = slim.conv2d(net, num_outputs=1, 
                                          kernel_size=[3, 3],
                                          activation_fn=tf.nn.sigmoid,
                                          scope='RefinedAlphaMatte')
        
        prediction_dict = {'alpha_matte': alpha_matte,
                           'refined_alpha_matte': refined_alpha_matte,
                           'trimaps': preprocessed_trimaps,}
        return prediction_dict
    
    def postprocess(self, prediction_dict, use_trimap=True):
        """Convert predicted output tensors to final forms.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.
                
        Returns:
            A dictionary containing the postprocessed results.
        """
        alpha_matte = prediction_dict.get('alpha_matte')
        refined_alpha_matte = prediction_dict.get('pred_refined_alpha_matte')
        if use_trimap:
            trimaps = prediction_dict.get('trimaps')
            alpha_matte = tf.where(tf.equal(trimaps, 128), alpha_matte,
                                   trimaps / 255.)
            refined_alpha_matte = tf.where(tf.equal(trimaps, 128),
                                           refined_alpha_matte,
                                           trimaps / 255.)
        postprocessed_dict = {'alpha_matte': alpha_matte,
                              'refined_alpha_matte': refined_alpha_matte}
        return postprocessed_dict
        
    
    def loss(self, prediction_dict, preprocessed_dict, epsilon=1e-12):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            preprocessed_dict: A dictionary of tensors holding groundtruth
                information, see preprocess function. The pixel values of 
                groundtruth_alpha_matte must be in [0, 128, 255].
                
        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        gt_images = preprocessed_dict.get('images')
        gt_fg = preprocessed_dict.get('images_fg')
        gt_bg = preprocessed_dict.get('images_bg')
        gt_alpha_matte = preprocessed_dict.get('alpha_mattes')
        alpha_matte = prediction_dict.get('alpha_matte')
        refined_alpha_matte = prediction_dict.get('refined_alpha_matte')
        pred_images = tf.multiply(alpha_matte, gt_fg) + tf.multiply(
            1 - alpha_matte, gt_bg)
        trimaps = prediction_dict.get('trimaps')
        weights = tf.where(tf.equal(trimaps, 128),
                           tf.ones_like(trimaps),
                           tf.zeros_like(trimaps))
        total_weights = tf.reduce_sum(weights) + epsilon
        first_stage_alpha_losses = tf.sqrt(
            tf.square(alpha_matte - gt_alpha_matte) + epsilon)
        first_stage_alpha_loss = tf.reduce_sum(
            first_stage_alpha_losses * weights) / total_weights
        first_stage_image_losses = tf.sqrt(
            tf.square(pred_images - gt_images) + epsilon) / 255.
        first_stage_image_loss = tf.reduce_sum(
            first_stage_image_losses * weights) / total_weights
        second_stage_alpha_losses = tf.sqrt(
            tf.square(refined_alpha_matte - gt_alpha_matte) + epsilon)
        second_stage_alpha_loss = tf.reduce_sum(
            second_stage_alpha_losses * weights) / total_weights
        loss = (self._first_stage_alpha_loss_weight * first_stage_alpha_loss +
                self._first_stage_image_loss_weight * first_stage_image_loss +
                self._second_stage_alpha_loss_weight * second_stage_alpha_loss)
        loss_dict = {'loss': loss}
        return loss_dict

