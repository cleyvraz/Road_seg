#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re
import cv2
import helper

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


def create_submission(submission_filename):
    """Create a submission files from the predictions images"""
    image_filenames = []
    for i in range(1, 51):
        image_filename = 'Datasets/test_set_images/test_' + str(i) + '/pred_' + str(i) + '.png'
        image_filenames.append(image_filename)
        masks_to_submission(submission_filename, *image_filenames)
    print("File '" + submission_filename + "' created")


def create_pred_images(predictions):
    """Create predictions images from predictions and save them in the same folder as the original picture"""
    
    #Number of patches per picture
    k = 1444
    
    #Dimensions of the test images
    w = 608
    h = 608
    
    patch_size = 16

    for i in range(1, 51):
        image_filename = 'Datasets/test_set_images/test_' + str(i) + '/pred_' + str(i) + '.png'
        offset = (i-1) * k
        gt_test = helper.label_to_img(w, h, patch_size, patch_size, predictions[offset: offset + k])
        gt_test *= 255
        cv2.imwrite(image_filename,gt_test)
