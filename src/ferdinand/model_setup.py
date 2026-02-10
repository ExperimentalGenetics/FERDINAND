
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def create_model(kernel_size = (3,3),
                 pool_size = (2,2),
                 first_filters = 32,
                 second_filters = 64,
                 third_filters = 128,
                 first_dense = 256,
                 second_dense = 128,
                 dropout_conv = 0.3,
                 dropout_dense = 0.3):

    model = Sequential()
    # First conv filters
    model.add(InputLayer(input_shape=(150, 150,1)))
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding="same"))
    model.add(Conv2D(first_filters, kernel_size, padding="same", activation = 'relu'))
    model.add(Conv2D(first_filters, kernel_size, padding="same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))

    # Second conv filter
    model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
    model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
    model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))

    # Third conv filter
    model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
    model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
    model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Flatten())

    # First dense
    model.add(Dense(first_dense, activation = "relu"))
    model.add(Dropout(dropout_dense))
    # Second dense
    model.add(Dense(second_dense, activation = "relu"))
    model.add(Dropout(dropout_dense))

    # Output layer
    model.add(Dense(360, activation="softmax"))
    model.summary()

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
