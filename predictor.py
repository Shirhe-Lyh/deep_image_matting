# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:49:09 2018

@author: shirhe-lyh
"""

import os
import tensorflow as tf


class Predictor(object):
    """Classify images to predifined classes."""
    
    def __init__(self,
                 frozen_inference_graph_path,
                 gpu_index=None,
                 gpu_memory_fraction=None):
        """Constructor.
        
        Args:
            frozen_inference_graph_path: Path to frozen inference graph.
            gpu_index: The GPU index to be used. Default None.
        """
        self._gpu_index = gpu_index
        self._gpu_memory_fraction = gpu_memory_fraction
        
        self._graph, self._sess = self._load_model(frozen_inference_graph_path)
        self._images = self._graph.get_tensor_by_name('image_tensor:0')
        self._trimaps = self._graph.get_tensor_by_name('trimap_tensor:0')
        self._alpha_mattes = self._graph.get_tensor_by_name('alpha_matte:0')
        self._refined_alpha_mattes = self._graph.get_tensor_by_name(
            'refined_alpha_matte:0')
        
    def _load_model(self, frozen_inference_graph_path):
        """Load a (frozen) Tensorflow model into memory.
        
        Args:
            frozen_inference_graph_path: Path to frozen inference graph.
            
        Returns:
            graph: A tensorflow Graph object.
            sess: A tensorflow Session object.
        
        Raises:
            ValueError: If frozen_inference_graph_path does not exist.
        """
        if not tf.gfile.Exists(frozen_inference_graph_path):
            raise ValueError('`frozen_inference_graph_path` does not exist.')
            
        # Specify which gpu to be used.
        if self._gpu_index is not None:
            if not isinstance(self._gpu_index, str):
                self._gpu_index = str(self._gpu_index)
            os.environ['CUDA_VISIBLE_DEVICES'] = self._gpu_index
        if self._gpu_memory_fraction is None:
            self._gpu_memory_fraction = 1.0
            
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_inference_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            
        config = tf.ConfigProto(allow_soft_placement = True) 
        config.gpu_options.per_process_gpu_memory_fraction = (
            self._gpu_memory_fraction)
        sess = tf.Session(graph=graph, config=config)
        return graph, sess
        
    def predict(self, images, trimaps):
        """Predict prediction tensors from inputs tensor.
        
        Args:
            preprocessed_inputs: A 4D float32 tensor with shape [batch_size, 
                height, width, channels] representing a batch of images.
            
        Returns:
            classes: A 1D integer tensor with shape [batch_size].
        """
        feed_dict = {self._images: images, self._trimaps: trimaps}
        [alpha_mattes, refined_alpha_mattes] = self._sess.run(
            [self._alpha_mattes, self._refined_alpha_mattes], 
            feed_dict=feed_dict)
        return alpha_mattes, refined_alpha_mattes
        