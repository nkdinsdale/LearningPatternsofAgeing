# Nicola Dinsdale 2019
# Main code for training vgg model
# Based on code by E Bluemke
########################################################################################################################
# Import dependencies
import numpy as np
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

batch_size = 2
nb_epochs = 200
kernel_size = 3
filters = 32

########################################################################################################################

if gender == 0:
    X_test = np.load('X_test_S0.npy')
    y_test = np.load('y_test_SO.npy')
elif gender == 1:
    X_test = np.load('X_test_S1.npy')
    y_test = np.load('y_test_S1.npy')
else:
    raise NotImplementedError('Gender must be 0 or 1')

# Reshape X to have the channels dimension
X_test = np.reshape(X_test, (-1, 128, 128, 20, 1))
print(X_test.shape)

model = Model()
model = vgg_model_orig(filters, im_height, im_width, number_slices, kernel_size)
model.load_weights( "model_checkpoint")

predicted_ages = model.predict(X_test, verbose=2)

np.save('y_predicted_1', predicted_ages)


model = Model()
model = vgg_model_orig(filters, im_height, im_width, number_slices, kernel_size)
model.load_weights( "model_checkpoint_2")

predicted_ages = model.predict(X_test, verbose=2)

np.save('y_predict_2', predicted_ages)

model = Model()
model = vgg_model_orig(filters, im_height, im_width, number_slices, kernel_size)
model.load_weights( "model_checkpoint_3")

predicted_ages = model.predict(X_test, verbose=2)

np.save('y_predicted_3', predicted_ages)