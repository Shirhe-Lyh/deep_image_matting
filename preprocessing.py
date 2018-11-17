# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:13:09 2018

@author: shirhe-lyh
"""

import tensorflow as tf


def random_crop(image, output_height, output_width, channels=3):
    """Crops the given image."""
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    image_aspect_ratio = tf.divide(width, height)
    output_aspect_ratio = tf.divide(output_width, output_height)
    
    def _random_crop_left_right():
        resized_width = tf.cast(
            tf.multiply(tf.to_float(height), output_aspect_ratio), 
            tf.int32)
        max_offset_width = width - resized_width + 1
        offset_width = tf.random_uniform(
            [], maxval=max_offset_width, dtype=tf.int32)
        return (tf.to_int32(tf.stack([0, offset_width, 0])),
                tf.to_int32(tf.stack([height, resized_width, channels])))
        
    def _random_crop_top_bottom():
        resized_height = tf.cast(
            tf.div(tf.to_float(width), output_aspect_ratio), 
            tf.int32)
        max_offset_height = height - resized_height + 1
        offset_height = tf.random_uniform(
            [], maxval=max_offset_height, dtype=tf.int32)
        return (tf.to_int32(tf.stack([offset_height, 0, 0])),
                tf.to_int32(tf.stack([resized_height, width, channels])))
    
    offsets, cropped_shape = tf.cond(
        tf.greater(image_aspect_ratio, output_aspect_ratio),
        _random_crop_left_right,
        _random_crop_top_bottom)
    image = tf.slice(image, offsets, cropped_shape)
    image = tf.squeeze(tf.image.resize_bilinear(
        [image], [output_height, output_width]), axis=0)
    return tf.reshape(image, [output_height, output_width, channels])


def random_crop_background(image, output_height, output_width, channels=3):
    """Crops the given image."""
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    cropped_height = tf.random_uniform([], minval=tf.div(height, 3), 
                                       maxval=height, dtype=tf.int32)
    cropped_width = tf.random_uniform([], minval=tf.div(width, 3), 
                                      maxval=width, dtype=tf.int32)
    max_offset_height = height - cropped_height + 1
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    max_offset_width = width - cropped_width + 1
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)
    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
    cropped_shape = tf.to_int32(
        tf.stack([cropped_height, cropped_width, channels]))
    image = tf.slice(image, offsets, cropped_shape)
    image = tf.squeeze(tf.image.resize_bilinear(
        [image], [output_height, output_width]), axis=0)
    return tf.reshape(image, [output_height, output_width, channels])
    

def _fixed_sides_resize(image, output_height, output_width, channels=3):
    """Resize images by fixed sides.
    
    Args:
        image: A 3-D image `Tensor`.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.

    Returns:
        resized_image: A 3-D tensor containing the resized image.
    """
    output_height = tf.convert_to_tensor(output_height, dtype=tf.int32)
    output_width = tf.convert_to_tensor(output_width, dtype=tf.int32)

    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_nearest_neighbor(
        image, [output_height, output_width], align_corners=False)
    resized_image = tf.squeeze(resized_image, axis=0)
    return tf.reshape(resized_image, [output_height, output_width, channels])


def _border_expand(image, mode='CONSTANT', constant_values=255):
    """Expands the given image.
    
    Args:
        Args:
        image: A 3-D image `Tensor`.
        output_height: The height of the image after Expanding.
        output_width: The width of the image after Expanding.
        resize: A boolean indicating whether to resize the expanded image
            to [output_height, output_width, channels] or not.

    Returns:
        expanded_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    
    def _pad_left_right():
        pad_left = tf.floordiv(height - width, 2)
        pad_right = height - width - pad_left
        return [[0, 0], [pad_left, pad_right], [0, 0]]
        
    def _pad_top_bottom():
        pad_top = tf.floordiv(width - height, 2)
        pad_bottom = width - height - pad_top
        return [[pad_top, pad_bottom], [0, 0], [0, 0]]
    
    paddings = tf.cond(tf.greater(height, width),
                       _pad_left_right,
                       _pad_top_bottom)
    expanded_image = tf.pad(image, paddings, mode=mode, 
                            constant_values=constant_values)
    return expanded_image


def border_expand(image, mode='CONSTANT', constant_values=255,
                  resize=False, output_height=None, output_width=None,
                  channels=3):
    """Expands (and resize) the given image."""
    expanded_image = _border_expand(image, mode, constant_values)
    if resize:
        if output_height is None or output_width is None:
            raise ValueError('`output_height` and `output_width` must be '
                             'specified in the resize case.')
        expanded_image = _fixed_sides_resize(expanded_image, output_height,
                                             output_width, channels)
        expanded_image.set_shape([output_height, output_width, channels])
    return expanded_image


def border_expand_and_resize(image, output_height, output_width, channels=3):
    """Border expand and resize."""
    num_ranks = tf.rank(image)
    image = tf.cond(tf.greater(num_ranks, 2), lambda: image, 
                lambda: tf.expand_dims(image, axis=2))
    values = tf.cast(
        tf.cond(tf.greater(num_ranks, 2), lambda: 255, lambda: 0),
        dtype=tf.uint8)
    return border_expand(
        image, constant_values=values, resize=True,
        output_height=output_height,
        output_width=output_width,
        channels=channels)


def preprocess_for_train(image_forground, image_background, alpha_matte, 
                         trimap, output_height, output_width):
    """Preprocessing for train."""
    preprocessed_image_fg = border_expand(
        image_forground, constant_values=255, resize=True,
        output_height=output_height, output_width=output_width, channels=3)
    alpha_matte = tf.expand_dims(alpha_matte, axis=2)
    preprocessed_alpha_matte = border_expand(
        alpha_matte, constant_values=0, resize=True,
        output_height=output_height, output_width=output_width, channels=1)
    trimap = tf.expand_dims(trimap, axis=2)
    preprocessed_trimap = border_expand(
        trimap, constant_values=0, resize=True,
        output_height=output_height, output_width=output_width, channels=1)
    preprocessed_image_bg = random_crop(
        image_background, output_height=output_height,
        output_width=output_width, channels=3)
    
    preprocessed_image_fg = tf.to_float(preprocessed_image_fg)
    preprocessed_alpha_matte = tf.to_float(preprocessed_alpha_matte)
    preprocessed_trimap = tf.to_float(preprocessed_trimap)
    preprocessed_image = tf.multiply(
        preprocessed_alpha_matte, preprocessed_image_fg) + tf.multiply(
            1 - preprocessed_alpha_matte, preprocessed_image_bg)
    preprocessed_dict = {'images_fg': preprocessed_image_fg,
                         'images_bg': preprocessed_image_bg,
                         'alpha_matte': preprocessed_alpha_matte,
                         'image': preprocessed_image,
                         'trimap': preprocessed_trimap}
    return preprocessed_dict
