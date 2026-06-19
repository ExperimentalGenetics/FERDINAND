
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

"""
Model-building utilities for X-ray angle prediction.

The module defines a convolutional neural network for angle classification and
metric helpers that compare predicted and true angles while respecting
wrap-around on a 360-degree circle.
"""

def create_model( 
        input_shape=(224, 224, 1),
        num_classes=360,
        dropout_rate=0.3,
        l2_reg=1e-4
):
    """
    Build the convolutional model used for angle classification.

    Parameters
    ----------
    input_shape : tuple[int, int, int], optional
        Shape of the input images as `(height, width, channels)`.
    num_classes : int, optional
        Number of output classes, typically one class per angle degree.
    dropout_rate : float, optional
        Dropout probability applied after pooling and dense blocks.
    l2_reg : float, optional
        L2 regularization coefficient for convolution and dense kernels.

    Returns
    -------
    tensorflow.keras.models.Model
        Uncompiled Keras model with a softmax output layer of size
        `num_classes`.
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
    Compute mean angular error between one-hot labels and predictions.

    Parameters
    ----------
    y_true : tensorflow.Tensor
        Ground-truth labels encoded as one-hot angle classes.
    y_pred : tensorflow.Tensor
        Predicted class probabilities or logits with the same class layout as
        `y_true`.

    Returns
    -------
    tensorflow.Tensor
        Scalar tensor containing the mean absolute angular error in degrees.
    """
    true_angles = tf.argmax(y_true, axis=-1)
    pred_angles = tf.argmax(y_pred, axis=-1)
    # Calculate the smallest difference between two angles, taking into account angular wrap-around
    diff = angle_difference(true_angles, pred_angles)
    return tf.reduce_mean(tf.abs(diff))

def angle_difference(x, y):
    """
    Compute the smallest circular difference between two angles.

    Parameters
    ----------
    x : tensorflow.Tensor | array-like
        First angle or tensor of angles in degrees.
    y : tensorflow.Tensor | array-like
        Second angle or tensor of angles in degrees.

    Returns
    -------
    tensorflow.Tensor | array-like
        Element-wise minimal angular difference in degrees on a 360-degree
        circle.
    """
    return 180 - abs(abs(x - y) - 180)
