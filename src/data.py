import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def random_crop_with_cutout(padsize=4, cutout_size=0):
    def func(x):
        xtmp = np.pad(x, pad_width=[(padsize,padsize),(padsize,padsize),(0,0)], mode="constant", constant_values=0)
        row, col, _ = xtmp.shape
        row_random, col_random = np.random.randint(-padsize, padsize, size=2)
        row_org, col_org, _ = x.shape
        c_row = row//2+row_random
        c_col = col//2+col_random
        xout = xtmp[c_row-row_org//2:c_row+row_org//2, c_col-col_org//2:c_col+col_org//2]

        # cutout
        if not cutout_size==0:
            top = np.random.randint(0-cutout_size//2, row_org-cutout_size//2)
            left = np.random.randint(0-cutout_size//2, col_org-cutout_size//2)
            bottom = top + cutout_size
            right = left + cutout_size
            if top < 0: top = 0
            if left < 0: left = 0
            xout[top:bottom, left:right].fill(xout.mean())

        return xout

    return func

def get_datagen_cutout(padsize=4, cutout_size=0):
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        preprocessing_function=random_crop_with_cutout(padsize=padsize, cutout_size=cutout_size))
    return datagen
