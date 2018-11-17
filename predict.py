# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:25:17 2018

@author: shirhe-lyh
"""

import cv2
import glob
import numpy as np
import os

import data_provider
import predictor


def border_expand(image, value=255, output_height=320, output_width=320):
    height, width = image.shape[:2]
    max_size = max(height, width)
    if len(image.shape) == 3:
        expanded_image = np.zeros((max_size, max_size, 3)) + value
    else:
        expanded_image = np.zeros((max_size, max_size)) + value
    if height > width:
        pad_left = (height - width) // 2
        expanded_image[:, pad_left:pad_left+width] = image
    else:
        pad_top = (width - height) // 2
        expanded_image[pad_top:pad_top+height] = image
    return expanded_image


if __name__ == '__main__':
    images_dir = '/data2/raycloud/deep_image_matting/test_images'
    frozen_inference_graph_path = ('./training/frozen_inference_graph_pb/' +
                                   'frozen_inference_graph.pb')
    
    matting_predictor = predictor.Predictor(frozen_inference_graph_path,
                                            gpu_index='1')
    
    for image_path in glob.glob(os.path.join(images_dir, '*.*')):
        image = cv2.imread(image_path)
        alpha = np.zeros(image.shape[:2])
        trimap = data_provider.trimap(alpha)
        
        #image = border_expand(image)
        #trimap = border_expand(trimap, value=0)
        
        images = np.expand_dims(image, axis=0)
        trimaps = np.expand_dims(trimap, axis=0)
        
        alpha_mattes, refined_alpha_mattes = matting_predictor.predict(
            images, trimaps)
        
        alpha_matte = np.squeeze(alpha_mattes, axis=0)
        refined_alpha_matte = np.squeeze(refined_alpha_mattes, axis=0)
        alpha_matte = 255 * alpha_matte
        refined_alpha_matte = 255 * refined_alpha_matte
        
        image_path_prefix = image_path.split('.')[0]
        output_path = image_path_prefix + '_alpha.jpg'
        output_path_refined = image_path_prefix + '_refined_alpha.jpg'
        cv2.imwrite(output_path, alpha_matte.astype(np.uint8))
        cv2.imwrite(output_path_refined, refined_alpha_matte.astype(np.uint8))
