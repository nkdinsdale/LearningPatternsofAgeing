# Nicola Dinsdale 2019
# Main code for training vgg model
# Based on code by E Bluemke
########################################################################################################################
# Import dependencies
from keras.models import Sequential, Model
from keras.layers import Input, BatchNormalization, Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.optimizers import RMSprop, Adam
########################################################################################################################

def vgg_model(filters, im_height, im_width, number_slices, kernel_size):
    input_shape = (im_height, im_width, number_slices, 1)
    inputTensor = Input(input_shape)

    x = Conv3D(filters, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same', name='conv_1',
               data_format="channels_last", trainable=True)(inputTensor)
    x = BatchNormalization()(x)
    x = Conv3D(filters, (kernel_size, kernel_size, kernel_size), activation='relu', name='conv_2', padding='same', trainable=True)(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2), padding='same', name='pool_1')(x)

    x = Conv3D(filters * 2, (kernel_size, kernel_size, kernel_size), activation='relu', name='conv_3', padding='same', trainable=True)(
        x)
    x = BatchNormalization()(x)
    x = Conv3D(filters * 2, (kernel_size, kernel_size, kernel_size), activation='relu', name='conv_4', padding='same', trainable=True)(
        x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2), padding='same', name='pool_2')(x)

    x = Conv3D(filters * 2, (kernel_size, kernel_size, kernel_size), activation='relu', name='conv_5', padding='same', trainable=True)(
        x)
    x = BatchNormalization()(x)
    x = Conv3D(filters * 2, (kernel_size, kernel_size, kernel_size), activation='relu', name='conv_6', padding='same', trainable=True)(
        x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2), padding='same', name='pool_3')(x)

    x = Conv3D(filters * 3, (kernel_size, kernel_size, kernel_size), activation='relu', name='conv_7', padding='same', trainable=True)(
        x)
    x = BatchNormalization()(x)
    x = Conv3D(filters * 3, (kernel_size, kernel_size, kernel_size), activation='relu', name='conv_8', padding='same', trainable=True)(
        x)
    x = BatchNormalization()(x)
    x = Conv3D(filters * 3, (kernel_size, kernel_size, kernel_size), activation='relu', name='conv_9', padding='same', trainable=True)(
        x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2), padding='same', name='pool_4')(x)

    x = Conv3D(filters * 3, (kernel_size, kernel_size, kernel_size), activation='relu', name='conv_10', padding='same', trainable=True)(
        x)
    x = BatchNormalization()(x)
    x = Conv3D(filters * 3, (kernel_size, kernel_size, kernel_size), activation='relu', name='conv_11', padding='same', trainable=True)(
        x)
    x = BatchNormalization()(x)
    x = Conv3D(filters * 3, (kernel_size, kernel_size, kernel_size), activation='relu', name='conv_12', padding='same')(
        x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2), padding='same', name='pool_5')(x)

    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(filters * 3, activation='relu', name='dense_1')(x)
    x = Dense(filters, activation='relu', name='dense_2')(x)
    x = Dropout(0.05)(x)
    prediction = Dense(1, activation='linear', name='dense_3')(x)

    rmsprop = RMSprop(lr=0.01)     #0.01
    age_regressor = Model(inputTensor, prediction)
    age_regressor.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['mse'])
    age_regressor.summary()

    return age_regressor

