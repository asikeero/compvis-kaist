import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_image_table(array, img_per_row, figsize=15, savename=''):
    """Create table of images based on data that has multiple images, one in each row"""
    d, N = array.shape
    assert N % img_per_row == 0
    width = 46
    height = 56
    img_per_column = N // img_per_row
    img_array = np.zeros((img_per_column * height, img_per_row * width), dtype=np.uint8)
    for col in range(img_per_column):
        for row in range(img_per_row):
            # add first row of pixels from first image to img_array, then first row of second pic etc.
            image_idx = col * img_per_row + row
            for pw in range(width):
                for ph in range(height):
                    img_array[col * height + ph, row * width + pw] = array[pw * height + ph, image_idx]
    
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_axis_off()
    ax.imshow(img_array, cmap='gray')
    if savename:
        plt.imsave(f'images/{savename}', img_array, cmap='gray')


def create_image(array, figsize=2, savename=''):
    """Shows the image, with a fixed size of 46x56 pixels"""
    width = 46
    height = 56
    assert len(array) == width * height, "Incorrect array size"
    img_array = np.zeros([height, width], dtype=np.uint8)
    for pw in range(width):
        for ph in range(height):
            img_array[ph, pw] = array[pw * height + ph]
    #img_array = np.reshape(array, (width, height)).T
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_axis_off()
    ax.imshow(img_array, cmap='gray')
    if savename:
        plt.imsave(f'images/{savename}', img_array, cmap='gray')

def upscale(array):
    """upscale to 0-255"""
    # make into 2d array if just one image is given
    if len(array.shape) != 2:
        array = np.expand_dims(array, axis=1)
    tmp = MinMaxScaler((0, 255)).fit_transform(array)
    return tmp
    
