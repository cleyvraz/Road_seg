import numpy as np
import albumentations as albu
import matplotlib.image as mpimg
from PIL import Image

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def binarize_predictions(predictions):
    predictions[np.where(predictions <= 0.5)] = 0
    predictions[np.where(predictions > 0.5)] = 1
    
    return predictions

def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def label_to_img_patches(labels):
    """To reconstituate an image after prediction.
       Can be use to process image after prediction (remove lonely patches)"""
    im = np.zeros([25, 25])#Nb of patches in a training image (400/16)
    idx = 0
    for i in range(0,25):
        for j in range(0,25):
            im[j, i] = labels[idx]
            idx = idx + 1
    return im

def label_to_img_patches_test(labels):
    """To reconstituate an image after prediction"""
    im = np.zeros([38, 38]) #nb of patches in test images (608/16)
    idx = 0
    for i in range(0,38):
        for j in range(0,38):
            im[j, i] = labels[idx]
            idx = idx + 1
    return im

def image_augmentation(image_size = 256, crop_prob = 0):
    """Returns an augmented image"""
    return albu.Compose([
        albu.RandomCrop(width = image_size, height = image_size, p=crop_prob),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.Transpose(p=0.5),], p = 1)

def feature_balancing(img_patches, gt_patches):
    """Takes the same number of patches which are road as background -> remove bias"""
    c0 = 0  # road
    c1 = 0  # bgrd
    for i in range(len(gt_patches)):
        if gt_patches[i] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print('Balancing training data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(gt_patches) if j == 1]
    idx1 = [i for i, j in enumerate(gt_patches) if j == 0]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(img_patches.shape)
    img_patches = img_patches[new_indices, :, :, :]
    gt_patches = gt_patches[new_indices]
    return img_patches, gt_patches
 


def list_neighbors(array,x,y):
    """Returns a list of the surrounding neighbors of pixel (x,y)."""
    n = []
    s = array.shape[0]-1 
    if((x == s) and (y ==s)): #Last corner. Somehow works with the other corner. Need to check better
        for i in range(-1,1):
            for j in range(-1,1):
                n.append(array[x+i][y+j])
    elif(x == s): #Last column
        for i in range(-1,1):
            for j in range(-1,2):
                n.append(array[x+i][y+j])
    elif(y == s): #Last row
        for i in range(-1,2):
            for j in range(-1,1):
                n.append(array[x+i][y+j])
    else:
        for i in range(-1,2):
            for j in range(-1,2):
                n.append(array[x+i][y+j])
    return n

def combine_surounded_patches(im):
    """Changes patch value to road if vertical/horizontal two direct neighbors are
    both road""" 
    s = im.shape[0]-1
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            if(im[x][y]==0):
                nb = list_neighbors(im,x,y)
                if((len(nb)== 9) and (nb[1]==1 and nb[7]==1)):#vetical road | middle of im
                    im[x][y] = 1
                if((len(nb) == 9) and (nb[3]==1 and nb[5]==1)):#horizontal road | middle of im
                    im[x][y] = 1
                if((len(nb)==6) and x==s and (nb[1]==1 and nb[5]==1)):
                    im[x][y] = 1
                if((len(nb)==6) and y==s and (nb[3]==1 and nb[5]==1)):
                    im[x][y] = 1
    return im

def remove_lonely_patches(im):
    """Switch patch value to road/background patches which are surrounded by background/road """
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            if(im[x][y]==1):
                nb = list_neighbors(im,x,y) 
                if(sum(nb)<4): #Nb of patches wich are road (=1) in the suroundings. May be interesting to switch number
                    im[x][y] = 0
            if(im[x][y]==0):
                nb = list_neighbors(im,x,y) 
                if(sum(nb)>5): 
                    im[x][y] = 1
    return im
         

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def value_to_class(v):
    foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
def patches_labelization(patches):
    labels = np.zeros(patches.shape[0], dtype = int)
    
    for i in range (patches.shape[0]):
        labels[i] = value_to_class(patches[i])
        
    return labels
