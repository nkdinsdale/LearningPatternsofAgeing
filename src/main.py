# Nicola Dinsdale 2019
# Main code for training vgg model
# Based on code by E Bluemke
########################################################################################################################
# Import dependencies
import numpy as np
import json
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential, Model

from model_factory import vgg_model
########################################################################################################################
gender = 0

if gender == 0:
    print( ' ************************************* ')
    print( ' ********* Print Gender = 0  ********* ')
    print( ' ************************************* ')
elif gender == 1:
    print( ' ************************************* ')
    print( ' ********* Print Gender = 1  ********* ')
    print( ' ************************************* ')
else:
    raise NotImplementedError('Gender must be 0 or 1')

########################################################################################################################
# Image variables
im_width = 128
im_height = 128
image_dimension = [im_width, im_height, 182]
number_slices = 20
number_subjects_0 = 7583
number_subjects_1 = 6917

batch_size = 8
nb_epochs = 200
kernel_size = 3
filters = 32
########################################################################################################################
# Load data
if gender == 0:
    X_train = np.load('X_train_S0.npy')
    y_train = np.load('y_train_SO.npy')
elif gender == 1:
    X_train = np.load('X_train_S1.npy')
    y_train = np.load('y_train_S1.npy')
else:
    raise NotImplementedError('Gender must be 0 or 1')

# Reshape X to have the channels dimension
X_train = np.reshape(X_train, (-1, 128, 128, 20, 1))
print(X_train.shape)
print(y_train.shape)

checkpoint_name = "model_checkpoint"
model_checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=40, verbose=1)
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1, min_delta=0.01, min_lr=0.000001)

print('Commencing model training...')

model = Model()
model = vgg_model(filters, im_height, im_width, slices, kernel_size)
history = model.fit(X_train, y_train,
                            batch_size = batch_size,
                            epochs = nb_epochs,
                            verbose = 2,
                            validation_split = 0.1,
                            shuffle = True,
                            callbacks = [model_checkpoint, early_stop, reduceLR])

weights_name = 'final_model'

model.save_weights(weights_name)

history = history.history
json.dump(history, open('history', 'w'))
