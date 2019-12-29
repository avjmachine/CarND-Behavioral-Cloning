import os
import csv
from math import ceil
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Lambda, Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import optimizers

# Setup the training data location here. The driving_log.csv and the IMG folder with all the training images should be in this folder
training_data_location = '../driving_data/run13/'

# Creating a variable to store the number of samples finally used for training and 2 empty arrays to store the training data, one for the image file path and one for the steering angle, as each row of the csv file is read

num_training_samples = 0
x_imagepaths = []
y_steerangles = []

# Introducing a correction factor for the steering angle when reading in images from the left and right camera
# This value is found by guesswork, and found to be giving good results
correction_camera_location = 0.2

with open(training_data_location + 'driving_log.csv') as csvfile:
    csvdata_reader = csv.reader(csvfile, delimiter=',')
    for i,row in enumerate(csvdata_reader):
        # the collected training data is biased with lot of straight driving data
        # this imbalance needs to be corrected by reducing the images with straight driving data
        # select images for training randomly with only 2.5% probability for steering angles between 0.02 and -0.02
        if (float(row[3]) < 0.02) and (float(row[3]) > -0.02):
            if np.random.random()<0.025:
                # for centre camera images
                x_imagepaths.append(training_data_location + 'IMG/' + os.path.split(row[0])[-1])
                y_steerangles.append(float(row[3]))
                # introduce steering angle correction for left camera images
                x_imagepaths.append(training_data_location + 'IMG/' + os.path.split(row[1])[-1])
                y_steerangles.append(float(row[3]) + correction_camera_location)
                # introduce steering angle correction for right camera images
                x_imagepaths.append(training_data_location + 'IMG/' + os.path.split(row[2])[-1])
                y_steerangles.append(float(row[3]) - correction_camera_location)
                # increment no. of training samples by three
                num_training_samples = num_training_samples + 3

        else:
            # select all images for other steering angles
            # for centre camera images
            x_imagepaths.append(training_data_location + 'IMG/' + os.path.split(row[0])[-1])
            y_steerangles.append(float(row[3]))
            # introduce steering angle correction for left camera images
            x_imagepaths.append(training_data_location + 'IMG/' + os.path.split(row[1])[-1])
            y_steerangles.append(float(row[3]) + correction_camera_location)
            # introduce steering angle correction for right camera images
            x_imagepaths.append(training_data_location + 'IMG/' + os.path.split(row[2])[-1])
            y_steerangles.append(float(row[3]) - correction_camera_location)
            # increment no. of training samples by three
            num_training_samples = num_training_samples + 3
        

# random shuffling of the dataset
x_imagepaths, y_steerangles = shuffle(x_imagepaths, y_steerangles)

# to show the distribution of the dataset
df = pd.DataFrame({'imagename':x_imagepaths, 'steer_angles':y_steerangles})
print(df.head())
df.hist(column="steer_angles", bins=ceil((df['steer_angles'].max()-df['steer_angles'].min())/0.05))
plt.show()

# to split dataset into training and validation sets
(x_train_imagepaths, x_valid_imagepaths, 
        y_train_steerangles, y_valid_steerangles) = train_test_split(x_imagepaths, y_steerangles, test_size=0.2)

# a generator to supply training images as well as augmented images to the model on the fly at run time
# this helps in cases where the entire training data set cannot be loaded at once due to memory constraints
def generator(x_data, y_data, batch_size):
    while True:
        for offset in range(0, len(x_data), batch_size):
            # Half of the batch size is used, since the data will be doubled in size, by augmentation using left-right flip of the images
            half_batch_size = batch_size//2
            halfbatch_x_imagepaths = x_data[offset : offset+half_batch_size]
            halfbatch_y_steerangles = np.array(y_data[offset : offset+half_batch_size])
        
            batch_x_images = np.zeros((len(halfbatch_x_imagepaths)*2, 80, 160, 3))
            batch_y_steerangles = np.zeros(len(halfbatch_y_steerangles)*2)
            #use this line if resizing not required - batch_x_images = np.zeros((len(batch_x_imagepaths), 160, 320, 3))
            
            for i in range(len(halfbatch_x_imagepaths)):
                img = mpimg.imread(halfbatch_x_imagepaths[i])
                # the images are resized to half their size, since it will save training time by reducing input pixels by a factor of 4
                batch_x_images[i] = cv2.resize(img, (160, 80), interpolation = cv2.INTER_AREA)
                # use this line if resizing not required - batch_x_images[i] = mpimg.imread(batch_x_imagepaths[i])
                batch_y_steerangles[i] = halfbatch_y_steerangles[i]
                
                # images are flipped along the vertical axis, to augment the data
                # the steering angle is changed to the opposite direction to match this flipping
                batch_x_images[2*i] = np.fliplr(batch_x_images[i])
                batch_y_steerangles[2*i] = -1 * halfbatch_y_steerangles[i]

            yield shuffle(batch_x_images, batch_y_steerangles)

# Definition of model architecture with Keras
model = Sequential()
# Images are cropped to remove the region above the horizon which are mostly not useful for prediction of steering angle
# Mostly the lower region showing the road is useful for the purpose of predicting the steering angle
model.add(Cropping2D(cropping=((25, 10), (0, 0)), input_shape=(80, 160, 3), data_format=None))
# use this line if image resizing not done - model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3), data_format=None))
# The image is normalized with 0 mean and 1 std. deviation to allow for easier training
model.add(Lambda(lambda x: (x/255.0) - 0.5))
# Convolutional layer to process the images and generate feature maps
model.add(Conv2D(6, (5,5), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(16, (5,5), activation='relu'))
# Flatten layer to be given as input for the fully connected layers
# Dropout layer added to reduce overfitting
model.add(Dropout(0.2))
model.add(Flatten())
# Fully connected layers with dropout layers to reduce overfitting
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(84, activation='relu'))
model.add(Dropout(0.5))
# Output layer with just one neuron outputting the steering angle
model.add(Dense(1))

# Adam optimizer used with following default parameters
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# Keras model compiled using a loss function of Mean Squared Error, since this is a regression problem
model.compile(adam, loss = 'mean_squared_error', metrics = '')

# Batch size for the training
batch_size = 32

# Model Checkpoint for Callbacks to store the best model with the lowest validation loss during training
# This helps prevent using an overfitted model, in case the validation loss goes up again after reaching a bottom
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)

# separate generators created for the training and validation datasets, not necessary, but just for clarity
training_generator = generator(x_train_imagepaths, y_train_steerangles, batch_size)
validation_generator = generator(x_valid_imagepaths, y_valid_steerangles, batch_size)

# Finally, training the model by supplying the data using generators and using callbacks to store the best result
model.fit_generator(training_generator, steps_per_epoch = ceil(num_training_samples/batch_size),
                    epochs = 15, verbose = 1, validation_data = validation_generator, 
                    callbacks = [mc], 
                    validation_steps = ceil(num_training_samples/batch_size))

print("Model saved!")
