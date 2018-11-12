# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:39:12 2018

@author: shirhe-lyh
"""

import glob
import numpy as np
import os


def random_select(image_bg_files, num_selects=20):
    if not isinstance(image_bg_files, np.ndarray):
        image_bg_files = np.array(image_bg_files)
    selected_files = np.random.choice(image_bg_files, size=num_selects)
    selected_files = [image_file.split('/')[-1] for image_file in 
                      selected_files]
    return selected_files


if __name__ == '__main__':
    image_fg_txt_path = './images_fg_without_boundary.txt'
    image_bg_dir = '/data2/raycloud/matting_bg'
    output_mapping_path = './fg_bg_mapping.txt'
    
    with open(image_fg_txt_path, 'r') as reader:
        image_fg_files = np.loadtxt(reader, str, delimiter='\n')
    image_bg_files = glob.glob(os.path.join(image_bg_dir, '*.*'))
    mapping_results = []
    for image_fg_file in image_fg_files:
        selected_bg_files = random_select(image_bg_files)
        for bg_file in selected_bg_files:
            mapping_results.append([image_fg_file, bg_file])
            
    with open(output_mapping_path, 'w') as writer:
        for image_fg_file, image_bg_file in mapping_results:
            writer.write(image_fg_file + '@' + image_bg_file + '\n')

