# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 18:46:37 2018

@author: shirhe-lyh
"""

import cv2
import glob
import numpy as np
import os


def without_boundary(image, boundary_width=50):
    if image is None:
        return False
    
    image = image[:, :, :3]
    top = np.sum(255 - image[:boundary_width])
    bottom = np.sum(255 - image[-boundary_width:])
    left = np.sum(255 - image[:, :boundary_width])
    right = np.sum(255 - image[:, -boundary_width:])
    if top < 1 and bottom and left < 1 and right < 1:
        return True
    return False


if __name__ == '__main__':
    images_root_dir = '/data2/raycloud/matting_resize'
    images_without_boundary = []
    for root, dirs, filenams in os.walk(images_root_dir):
        count = 0
        for dir_name in dirs:
            count += 1
            print('{}/{}: {}'.format(count, len(dirs), dir_name))
            image_files = glob.glob(os.path.join(dir_name, '*.png'))
            for image_file in image_files:
                image = cv2.imread(image_file, -1)
                if without_boundary(image):
                    images_without_boundary.append(image_file)
            
    print('Total selected images: ', len(images_without_boundary))
    
    with open('images_fg_without_boundary.txt', 'w') as writer:
        for image_file in images_without_boundary:
            writer.write(image_file + '\n')
