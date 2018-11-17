#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 21:44:24 2018

@author: shirhe-lyh
"""

import cv2
import glob
import numpy as np
import os
import tensorflow as tf

from object_detection.utils import ops as utils_ops

flags = tf.app.flags

flags.DEFINE_string('images_dir', 
                    './datasets/images',
                    'Path to images directory.')
flags.DEFINE_string('output_dir', 
                    './datasets/trimaps', 
                    'Path to the output trimap`s directory.')
flags.DEFINE_string('frozen_inference_graph_path', 
                    None, 
                    'Path to the inference grpah .pb file.')

FLAGS = flags.FLAGS


class Predictor(object):
    """Predict mask."""
    
    def __init__(self, 
                 frozen_inference_graph_path,
                 detection_score_threshold=0.97, 
                 detection_mask_threshold=0.9):
        """Constructor."""
        self._detection_score_threshold = detection_score_threshold
        self._detection_mask_threshold = detection_mask_threshold
        
        self._graph, self._sess = self._load_model(frozen_inference_graph_path)
        self._image_tensor = self._graph.get_tensor_by_name('image_tensor:0')
        self._tensor_dict = self._get_tensors()
        
    def _load_model(self, frozen_inference_graph_path):
        """Load a (frozen) inference graph to memory."""
        if not tf.gfile.Exists(frozen_inference_graph_path):
            raise ValueError('`frozen_inference_graph_path` does not exist.')
            
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_inference_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
        sess = tf.Session(graph=detection_graph)
        return detection_graph, sess
    
    def _get_tensors(self, shape=[480, 480]):
        # Get handles to input and output tensors
        ops = self._graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self._graph.get_tensor_by_name(
                    tensor_name)
                
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(
                tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(
                tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates 
            # to image coordinates and fit the image size.
            real_num_detection = tf.cast(
                tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(
                detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(
                detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, shape[0], shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 
                           self._detection_mask_threshold), 
                tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
        return tensor_dict
    
    def predict(self, image):
        """Predict boundingboxes and masks."""
        if image is None:
            return None
            
        # Run inference
        feed_dict = {self._image_tensor: np.expand_dims(image, axis=0)}
        output_dict = self._sess.run(self._tensor_dict, feed_dict=feed_dict)
        # all outputs are float32 numpy arrays, so convert types as 
        # appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict
    
    def postprocess(self, prediction_dict):
        """Postprocessing."""
        masks = prediction_dict.get('detection_masks', None)
        if masks is not None:
            return masks[0]
        return None
    
    def close_seesion(self):
        self._sess.close()
        
        
if __name__ == '__main__':
    images_dir = FLAGS.images_dir
    output_dir = FLAGS.output_dir
    frozen_inference_graph_path = FLAGS.frozen_inference_graph_path
      
    mask_predictor = Predictor(frozen_inference_graph_path)
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    count = 0
    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    for image_path in image_paths:
        count += 1
        if count % 100 == 0:
            print('{}/{}'.format(count, len(image_paths)))
            
        image_name = image_path.split('/')[-1]
        output_path = os.path.join(output_dir, image_name)
        if os.path.exists(output_path):
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print('%s does not exist.' % image_path)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        prediction_dict = mask_predictor.predict(image)
        mask = mask_predictor.postprocess(prediction_dict)
        if mask is None:
            print('No mask: %s' % image_path)
            continue
        
        cv2.imwrite(output_path, mask)
        
    mask_predictor.close_seesion()
    
    