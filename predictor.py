# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:49:09 2018

@author: shirhe-lyh
"""

import cv2
import numpy as np
import os
import tensorflow as tf
import urllib

#from timeit import default_timer as timer


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
        self._alpha_mattes = self._graph.get_tensor_by_name('alpha_matte_1:0')
        self._refined_alpha_mattes = self._graph.get_tensor_by_name(
            'refined_alpha_matte_1:0')
        
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
            os.environ["CUDA_VISIBLE_DEVICES"] = self._gpu_index
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
        config.gpu_options.per_process_gpu_memory_fraction = 0.50
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
    
#    def predict_from_urls(self, image_urls=None):
#        """Predict prediction tensors from image urls.
#        
#        Args:
#            image_urls: A list containing image urls.
#            
#        Returns:
#            classes_dict: A dictionary containing class labels for each image.
#            
#        Raises:
#            ValueError: If image_urls is None.
#        """
##        t_start_download = timer()
#        images_dict = download_images(image_urls)
##        t_end_download = timer()
##        print('download image time: ', t_end_download - t_start_download)
#        
#        classes_dict = {}
#        valid_image_urls = []
#        valid_images = []
#        for image_url, image in images_dict.items():
#            if image is None:
#                classes_dict[image_url] = []
#                continue
#            valid_image_urls.append(image_url)
#            valid_images.append(image)
#        if not valid_images:
#            return classes_dict
#        
##        t_start_predict = timer()
#        valid_classes = self.predict(valid_images)
##        print('predict time: ', timer() - t_start_predict)
#        for image_url, class_label in zip(valid_image_urls, valid_classes):
#            classes_dict[image_url] = [int(class_label)]
#        return classes_dict
#    
#    
#def download_images(image_urls=None):
#    """Download images.
#    
#    Args:
#        image_urls: A list containing image urls.
#        
#    Returns:
#        images_dict: A dictionary representing a batch of images.
#        
#    Raises:
#        ValueError: If image_urls is None.
#    """
#    if image_urls is None:
#        raise ValueError('`image_urls` must be specified.')
#        
#    images_dict = {}
#    for image_url in image_urls:
#        try:
#            # For python2
##            req_image_url = urllib.urlopen(image_url)
#            # For python3
#            req_image_url = urllib.request.urlopen(image_url)
#        except:
#            print(image_url + ' does not exist.')
#            images_dict[image_url] = None
#            continue
#        image = np.asarray(bytearray(req_image_url.read()), dtype='uint8')
#        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#        images_dict[image_url] = image
#    return images_dict
        