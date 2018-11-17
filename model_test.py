# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:29:50 2018

@author: shirhe-lyh
"""

import cv2
import numpy as  np
import tensorflow as tf

import data_provider
import model


def alpha_matte(image):
    if image.shape[2] > 3:
        alpha = image[:, :, 3]
        alpha = np.where(alpha > 0, 1, 0)
    else:
        alpha = np.ones(image.shape[:2])
        diff_values = 255 - image
        diff_value = np.sum(np.abs(diff_values), axis=2)
        alpha = np.where(diff_value > 0, alpha, 0)
    alpha = alpha.astype(np.uint8)
    return alpha


def image_foreground(image, alpha_matte):
    channels = np.split(image, 3, axis=2)
    channels_fg = []
    alpha_matte_expanded = np.expand_dims(alpha_matte, axis=2)
    for channel in channels:
        channel_fg = channel * alpha_matte_expanded
        channel_fg = np.where(channel_fg > 0, channel_fg, 255)
        channels_fg.append(channel_fg)
    image_fg = np.concatenate(channels_fg, axis=2)
    return image_fg


if __name__ == '__main__':
#    image_path = './IMG_5846.png'
    image_path = './2-2.png'
    image = cv2.imread(image_path, -1)
    alpha = alpha_matte(image)
    image = image[:, :, :3]
    #image = image_foreground(image, alpha)
    trimap = data_provider.trimap(alpha)
    
    image_bg_path = './bg.jpg'
    image_bg = cv2.imread(image_bg_path)
    
#    cv2.imshow('image', image)
#    cv2.imshow('alpha', alpha * 255)
#    cv2.imshow('image_bg', image_bg)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    matting_model = model.Model(is_training=True)
    expanded_images_fg = np.expand_dims(image, axis=0)
    expanded_images_bg = np.expand_dims(image_bg, axis=0)
    expanded_alphas = np.expand_dims(alpha, axis=0)
    expanded_trimaps = np.expand_dims(trimap, axis=0)
    preprocessed_dict = matting_model.preprocess(expanded_trimaps,
                                                 None,
                                                 expanded_images_fg, 
                                                 expanded_images_bg, 
                                                 expanded_alphas)
    preprocessed_images = preprocessed_dict.get('images')
    preprocessed_images_fg = preprocessed_dict.get('images_fg')
    preprocessed_images_bg = preprocessed_dict.get('images_bg')
    preprocessed_alphas = preprocessed_dict.get('alpha_mattes')
    preprocessed_trimaps = preprocessed_dict.get('trimaps')
    prediction_dict = matting_model.predict(preprocessed_dict)
    alpha_pred = prediction_dict.get('alpha_matte')
    alpha_refine_pred = prediction_dict.get('refined_alpha_matte')
    postprocessed_dict = matting_model.postprocess(prediction_dict)
    postprocessed_alpha = postprocessed_dict.get('alpha_matte')
    postprocessed_r_alpha = postprocessed_dict.get('refined_alpha_matte')
    loss_dict = matting_model.loss(prediction_dict, preprocessed_dict)
    loss = loss_dict.get('loss')
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        (pre_images, pre_images_fg, 
         pre_images_bg, pre_alphas, 
         pre_trimaps, 
         p_alphas, p_r_alphas, 
         post_alphas, post_r_alphas,
         loss_) = sess.run(
            [preprocessed_images, preprocessed_images_fg,
             preprocessed_images_bg, preprocessed_alphas,
             preprocessed_trimaps,  # Preprocess, success
             alpha_pred, alpha_refine_pred,  # Predict, success
             postprocessed_alpha, postprocessed_r_alpha,  # Postprocess, success
             loss  # Loss, success
             ]) 
    
    pre_image = np.squeeze(pre_images).astype(np.uint8)
    pre_image_fg = np.squeeze(pre_images_fg).astype(np.uint8)
    pre_image_bg = np.squeeze(pre_images_bg).astype(np.uint8)
    pre_alpha = np.squeeze(pre_alphas).astype(np.uint8)
    pre_trimap = np.squeeze(pre_trimaps).astype(np.uint8)
    p_alpha = np.squeeze(p_alphas).astype(np.uint8)
    p_r_alpha = np.squeeze(p_r_alphas).astype(np.uint8)
    post_alpha = np.squeeze(post_alphas).astype(np.uint8)
    post_r_alpha = np.squeeze(post_r_alphas).astype(np.uint8)
    
    print('Loss: ', loss_)
    
    cv2.imshow('pre_image', pre_image)
    cv2.imshow('pre_image_fg', pre_image_fg)
    cv2.imshow('pre_image_bg', pre_image_bg)
    cv2.imshow('pre_alpha', pre_alpha * 255)
    cv2.imshow('pre_trimap', pre_trimap)
    cv2.imshow('p_alpha', p_alpha * 255)
    cv2.imshow('p_r_alpha', p_r_alpha * 255)
    cv2.imshow('post_alpha', post_alpha * 255)
    cv2.imshow('post_r_alpha', post_r_alpha * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
