
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Add
from tensorflow.keras.layers import Model
from tensorflow.keras.layers import l2

"""
This module defines the CNN architecture for predicting angles from X-ray images. 
The model is designed to handle the specific characteristics of the dataset, including the input image size and the number of output classes (angles). 
"""

def create_model( 
        input_shape=(224, 224, 1),
        num_classes=360,
        dropout_rate=0.3,
        l2_reg=1e-4
):
    """
    Create a CNN model for angle prediction from X-ray images.
    :param input_shape: Shape of the input images (height, width, channels)
    :param num_classes: Number of output classes (angles)
    :param dropout_rate: Dropout rate for regularization
    :param l2_reg: L2 regularization coefficient
    :return: Compiled Keras model
    """
    inputs = Input(shape=input_shape)

    # initial conv block
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    # second conv block with residual connection
    shortcut = Conv2D(64, (1, 1), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    # third conv block with residual connection
    shortcut = Conv2D(128, (1, 1), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    # fourth conv block with residual connection
    shortcut = Conv2D(256, (1, 1), padding='same')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(dropout_rate)(x)

    # global average pooling (reduces overfitting compared to Flatten)
    x = GlobalAveragePooling2D()(x)

    # dense layers with batch normalization
    x = Dense(512, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(256, kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    # output layer - SOFTMAX for multi-class classification
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def angle_error(y_true, y_pred):
    """
    Calculate the mean difference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector (one-hot encoded).
    """
    true_angles = tf.argmax(y_true, axis=-1)
    pred_angles = tf.argmax(y_pred, axis=-1)
    # Calculate the smallest difference between two angles, taking into account angular wrap-around
    diff = angle_difference(true_angles, pred_angles)
    return tf.reduce_mean(tf.abs(diff))

def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)
