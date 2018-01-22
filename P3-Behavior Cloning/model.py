import csv
import numpy as np
from scipy.misc import imread

import warnings
warnings.filterwarnings('ignore')

'''
Reading csv file and loading the images and steering angle into numpy
'''
def load_path(filename):
    print('Loading data ...')
    imgs = []
    angles = []
    with open(filename) as csvfile:
        # handling with column headers if exist
        has_header = csv.Sniffer().has_header(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.reader(csvfile)
        if has_header: next(reader)
        for center_name, left_name, right_name, angle, throttle, brake, speed in reader:
            # reading center image and angle
            center_name = 'data/'+center_name
            center_img = imread(center_name, mode='RGB')
            imgs.append(center_img)
            angles.append(float(angle))
            # flipping center image for handling class distribution
            imgs.append(np.fliplr(center_img))
            angles.append(-float(angle))
            # storing left image and adding 0.2 offset to steering angle
            left_name = 'data/' + left_name.strip()
            imgs.append(imread(left_name, mode='RGB'))
            angles.append(float(angle) + 0.2)
            # storing right image and subtracting 0.2 offset to steering angle
            right_name = 'data/' + right_name.strip()
            imgs.append(imread(right_name, mode='RGB'))
            angles.append(float(angle) - 0.2)

    print('Finished loading data',len(imgs), len(angles))
    return np.array(imgs), np.array(angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, MaxPooling2D, Activation, Dropout, Lambda, Convolution2D, Cropping2D, ELU, BatchNormalization

'''
Steering model developed by comma.ai
https://github.com/commaai/research
'''
def comma_model():
    input_shape = (160, 320, 3)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(16,8,8,subsample=(4,4), border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(32,5,5,subsample=(2,2),border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(64,5,5,subsample=(2,2),border_mode='same'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

if __name__ == '__main__':
    X, y = load_path('data/driving_log.csv')
    model = comma_model()
    model.fit(X, y, validation_split=0.2, shuffle=True, epochs=5)
    model.save('model.h5')
