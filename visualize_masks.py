#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 22:47:38 2018

@author: shirhe-lyh
"""

import cv2
import glob
import numpy as np
import os


if __name__ == '__main__':
    masks_dir = './datasets/trimaps'
    output_dir = './datasets/trimaps_recgonized'
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for mask_path in glob.glob(os.path.join(masks_dir, '*.*')):
        mask = cv2.imread(mask_path, 0) * 255
        mask_name = mask_path.split('/')[-1]
        if np.sum(mask) < 1:
            print('invalid mask: %s' % mask_name)
        output_path = os.path.join(output_dir, mask_name)
        cv2.imwrite(output_path, mask)
        