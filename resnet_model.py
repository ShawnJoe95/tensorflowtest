

import keras


# 程序没有管每个节点的命名, 主路1,3,1结构,侧路1结构, ResNet中的block
def conv_block(input_tensor, filters):
    filter1, filter2, filter3 = filters

    x = keras.layers.Conv2D(filter1,(1,1),strides=1)(input_tensor)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filter2,(3,3),strides=1,padding='same')(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filter3,(1,1),strides=1)(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)

    y = keras.layers.Conv2D(filter3,(1,1),strides=1)(input_tensor)
    y = keras.layers.BatchNormalization(axis=-1)(y)

    # out = keras.layers.merge([x,y],mode='sum')
    x = keras.layers.Add()([x, y])
    x = keras.layers.Activation('relu')(x)

    return x


def identity_block(input_tensor, filters):
    filter1, filter2, filter3 = filters

    x = keras.layers.Conv2D(filter1,(1,1),strides=1)(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filter2,(3,3),strides=1,padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filter3,(1,1),strides=1)(x)
    x = keras.layers.BatchNormalization()(x)

    # identity block 和 conv block 差别在于侧路（input）需不需要卷积
    x = keras.layers.Add()([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x


def resnet_model(out_class, input_shape): # ResNet50
    inputs = keras.layers.Input(shape=input_shape) #1,3,224,224

    x = keras.layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs) #conv1  1,64,112,112
    x = keras.layers.BatchNormalization(axis=-1)(x) #bn_conv1
    x = keras.layers.Activation('relu')(x) #conv1_relu

    x = keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same')(x) # 1,64,56,56

    # block1[3]  (64,64,256) 1,2 in:1,64,56,56
    x = conv_block(x, [64, 64, 256]) #out=1,256,56,56
    x = identity_block(x, [64, 64, 256]) #out=1,256,56,56
    x = identity_block(x, [64, 64, 256]) #out=1,256,56,56

    # block2[4]  (128,128,512) 1,3 in:1,256,56,56
    x = conv_block(x, [128,128,512]) #out=1,512,28,28
    x = identity_block(x, [128,128,512]) #out=1,512,28,28
    x = identity_block(x, [128,128,512]) #out=1,512,28,28
    x = identity_block(x, [128, 128, 512])  # out=1,512,28,28

    # block3[6] (256,256,1024) 1,5 in:1,512,28,28
    x = conv_block(x, [256,256,1024])  # out=1,1024,14,14
    x = identity_block(x, [256, 256, 1024])  # out=1,1024,14,14
    x = identity_block(x, [256, 256, 1024])  # out=1,1024,14,14
    x = identity_block(x, [256, 256, 1024])  # out=1,1024,14,14
    x = identity_block(x, [256, 256, 1024])  # out=1,1024,14,14
    x = identity_block(x, [256, 256, 1024])  # out=1,1024,14,14

    # block4[3] (512,512,2048) 1,2 in:1,1024,14,14
    x = conv_block(x, [512,512,2048])  # out=1,2048,7,7
    x = identity_block(x, [512, 512, 2048])  # out=1,2048,7,7
    x = identity_block(x, [512, 512, 2048])  # out=1,2048,7,7

    # Averagepool kernel_size=7, stride=1 out=1,2048,1,1
    x = keras.layers.AveragePooling2D(pool_size=(7, 7), strides=1)(x)

    # flatten
    x = keras.layers.Flatten()(x)

    # # Dense
    # x = Dense(1000)(x) # out=1,1000

    # Dense,这里改造了一下，适应cifar10
    x = keras.layers.Dense(out_class)(x)  # out=1,1000

    out = keras.layers.Activation('softmax')(x)

    model = keras.models.Model(inputs=inputs, outputs=out)

    return model
