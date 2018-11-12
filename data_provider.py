# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:44:07 2018

@author: shirhe-lyh
"""

import cv2
import numpy as np
import os


def alpha_matte(image):
    """Returns the alpha channel of a given image."""
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


def trimap(alpha_matte, mode='boundary', boundary_width=50):
    """Returns the trimap of a given alpha matte."""
    if mode == 'trivial':
        return np.ones_like(alpha_matte, dtype=np.uint8) * 128
    elif mode == 'boundary':
        trimap_b = np.ones_like(alpha_matte, dtype=np.uint8) * 128
        trimap_b[:boundary_width] = 0
        trimap_b[-boundary_width:] = 0
        trimap_b[:, :boundary_width] = 0
        trimap_b[:, -boundary_width:] = 0
        return trimap_b
    
    erode_kernel_size = np.random.randint(30, 100)
    dilate_kernel_size = np.random.randint(3, 100)
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (erode_kernel_size, erode_kernel_size))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
    eroded_alpha = cv2.erode(alpha_matte, erode_kernel)
    dilated_alpha = cv2.dilate(alpha_matte, dilate_kernel)
    
    trimap_d = np.where(dilated_alpha > 0, 128, 0)
    trimap_e = np.where(eroded_alpha > 0, 127, 0)
    trimap_sum = trimap_d + trimap_e
    trimap_sum = trimap_sum.astype(np.uint8)
    return trimap_sum


def provide(txt_path, images_fg_dir=None, images_bg_dir=None):
    """Returns the paths of images."""
    if not os.path.exists(txt_path):
        raise ValueError('`txt_path` does not exist.')
        
    with open(txt_path, 'r') as reader:
        txt_content = np.loadtxt(reader, str, delimiter='@')
        np.random.shuffle(txt_content)
    if images_fg_dir is None and images_bg_dir is None:
        return txt_content
    image_paths = []
    for image_fg_rel_path, image_bg_rel_path in txt_content:
        image_fg_abs_path = image_fg_rel_path
        image_bg_abs_path = image_bg_rel_path
        if images_fg_dir is not None:
            image_fg_abs_path = os.path.join(images_fg_dir, image_fg_rel_path)
        if images_bg_dir is not None:
            image_bg_abs_path = os.path.join(images_bg_dir, image_bg_rel_path)
        image_paths.append([image_fg_abs_path, image_bg_abs_path])
    return image_paths


if __name__ == '__main__':
    image_path = './IMG_5846.png'
    image = cv2.imread(image_path, -1)
    alpha = alpha_matte(image)
    image = image[:, :, :3]
    trimap_random = trimap(alpha)
    trimap_trivial = trimap(alpha, mode='trivial')
    
    cv2.imshow('image', image)
    cv2.imshow('alpha', alpha * 255)
    cv2.imshow('trimap', trimap_random.astype(np.uint8))
    cv2.imshow('trimap_trivial', trimap_trivial)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

