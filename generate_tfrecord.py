#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 09:02:10 2018

@author: shirhe-lyh
"""

"""Generate tfrecord file from images.

Example Usage:
---------------
python3 train.py \
    --images_dir: Path to images (directory).
    --annotation_path: Path to annotatio's .txt file.
    --output_path: Path to .record.
    --resize_side_size: Resize images to fixed size.
"""

import cv2
import io
import PIL.Image
import tensorflow as tf

import data_provider

flags = tf.app.flags

flags.DEFINE_string('images_fg_dir', 
                    './datasets/images',
                    'Path to images (directory).')
flags.DEFINE_string('images_bg_dir', 
                    './datasets/images_bg',
                    'Path to images (directory).')
flags.DEFINE_string('trimaps_dir', 
                    './datasets/trimaps',
                    'Path to images (directory).')
flags.DEFINE_string('annotation_path', 
                    './datasets/images_correspondence.txt',
                    'Path to fg_bg_mapping`s .txt file.')
flags.DEFINE_string('output_path', 
                    './datasets/train.record',
                    'Path to output tfrecord file.')
flags.DEFINE_integer('resize_side_size', 320, 'Resize images to fixed size.')

FLAGS = flags.FLAGS


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(image_fg_path, image_bg_path, trimap_path,
                      resize_size=None):
    image_fg = cv2.imread(image_fg_path, -1)
    if image_fg is None:
        print(image_fg_path)
        return None
    #image_fg = cv2.imdecode(np.fromfile(image_fg_path, dtype=np.uint8), -1)
    alpha = data_provider.alpha_matte(image_fg)
    trimap = cv2.imread(trimap_path, 0)
    trimap_b = data_provider.trimap(trimap)
    image_fg = image_fg[:, :, :3]
    image_bg = cv2.imread(image_bg_path)
    
    height, width, _ = image_fg.shape
    
    # Resize
    if resize_size is not None:
        if width > height:
            width = int(width * resize_size / height)
            height = resize_size
        else:
            width = resize_size
            height = int(height * resize_size / width)
        image_fg = cv2.resize(image_fg, (width, height))
        alpha = cv2.resize(alpha, (width, height))
        trimap_b = cv2.resize(trimap_b, (width, height))
    else:
        image_bg = cv2.resize(image_bg, (width, height))
        
    # Encode
    pil_image_fg = PIL.Image.fromarray(image_fg)
    bytes_io = io.BytesIO()
    pil_image_fg.save(bytes_io, format='JPEG')
    encoded_fg = bytes_io.getvalue()
    pil_image_bg = PIL.Image.fromarray(image_bg)
    bytes_io = io.BytesIO()
    pil_image_bg.save(bytes_io, format='JPEG')
    encoded_bg = bytes_io.getvalue()
    pil_trimap = PIL.Image.fromarray(trimap_b)
    bytes_io = io.BytesIO()
    pil_trimap.save(bytes_io, format='JPEG')
    encoded_trimap = bytes_io.getvalue()
    pil_alpha = PIL.Image.fromarray(alpha)
    bytes_io = io.BytesIO()
    pil_alpha.save(bytes_io, format='JPEG')
    encoded_alpha = bytes_io.getvalue()
    
    tf_example = tf.train.Example(
        features=tf.train.Features(feature={
            'image_fg/encoded': bytes_feature(encoded_fg),
            'image/format': bytes_feature('jpg'.encode()),
            'image_bg/encoded': bytes_feature(encoded_bg),
            'trimap/encoded': bytes_feature(encoded_trimap),
            'alpha_matte/encoded': bytes_feature(encoded_alpha),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width)}))
    return tf_example


def generate_tfrecord(image_paths, output_path, resize_size=None):
    num_valid_tf_example = 0
    writer = tf.python_io.TFRecordWriter(output_path)
    for image_fg_path, image_bg_path, trimap_path in image_paths:
        if not tf.gfile.GFile(image_fg_path):
            print('%s does not exist.' % image_fg_path)
            continue
        if not tf.gfile.GFile(image_bg_path):
            print('%s does not exist.' % image_bg_path)
            continue
        tf_example = create_tf_example(image_fg_path, image_bg_path, 
                                       trimap_path, resize_size)
        if tf_example is None:
            continue
        writer.write(tf_example.SerializeToString())
        num_valid_tf_example += 1
        
        if num_valid_tf_example % 100 == 0:
            print('Create %d TF_Example.' % num_valid_tf_example)
            
    writer.close()
    print('Total create TF_Example: %d' % num_valid_tf_example)
    print('The number of skiped images: %d' % (len(image_paths) - 
                                               num_valid_tf_example))
    
    
def main(_):
    images_fg_dir = FLAGS.images_fg_dir
    images_bg_dir = FLAGS.images_bg_dir
    trimaps_dir = FLAGS.trimaps_dir
    annotation_path = FLAGS.annotation_path
    record_path = FLAGS.output_path
    resize_size = FLAGS.resize_side_size
    
    image_paths = data_provider.provide(annotation_path, images_fg_dir,
                                        images_bg_dir, trimaps_dir)
    
    generate_tfrecord(image_paths, record_path, resize_size)
    
    
if __name__ == '__main__':
    tf.app.run()