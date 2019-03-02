import numpy as np
import keras.models
from scipy.misc import imread, imresize,imshow
import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def init():
    num_classes = 47
    img_size = 28
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    # model.add(keras.layers.Reshape((img_size,img_size,1), input_shape=(784,)))
    model.add(keras.layers.Conv2D(filters=12, kernel_size=(5,5), strides=2, activation='relu', 
                                  input_shape=(img_size,img_size,1)))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(.5))
    
    model.add(keras.layers.Conv2D(filters=18, kernel_size=(3,3) , strides=2, activation='relu'))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(.5))
    
    model.add(keras.layers.Conv2D(filters=24, kernel_size=(2,2), activation='relu'))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    
    # model.add(keras.layers.Conv2D(filters=30, kernel_size=(3,3), activation='relu'))
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=150, activation='relu'))
    model.add(keras.layers.Dense(units=num_classes, activation='softmax'))
    
    #load woeights into new model
    model.load_weights("weights.h5")
    print("Loaded Model from disk")

    #compile and evaluate loaded model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    #loss,accuracy = model.evaluate(X_test,y_test)
    #print('loss:', loss)
    #print('accuracy:', accuracy)
    graph = tf.get_default_graph()

    return model, graph