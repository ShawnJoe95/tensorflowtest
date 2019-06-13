import keras
from keras.layers import Conv2D,MaxPooling2D,Activation,BatchNormalization

def conv_block(input_tensor, filters):
    filter1, filter2 = filters

    x = Conv2D(filter1,(1,1),strides=1)(input_tensor)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (1, 1), strides=1)(x)
    x = BatchNormalization(axis=-1)(x)


    x = Activation('relu')(x)


    y = Conv2D(filter2, (1, 1), strides=1)(input_tensor)
    y = keras.layers.BatchNormalization(axis=-1)(y)
    y = keras.layers.Activation('relu')(y)
