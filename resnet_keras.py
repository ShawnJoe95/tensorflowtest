import keras
from resnet_model import resnet_model
from keras.datasets import cifar10
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import math

if __name__ == '__main__':

    n_class = 10
    img_w = 32
    img_h = 32
    BATCH_SIZE = 128
    EPOCH = 3

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32')
    x_train /= 255.
    y_train = keras.utils.np_utils.to_categorical(y_train, n_class)

    x_test = x_test.astype('float32')
    x_test /= 255.
    y_test = keras.utils.np_utils.to_categorical(y_test, n_class)


    tb = TensorBoard(log_dir='log')
    cp = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',save_best_only=1, mode='auto')


    def step_decay(epoch):
        initial_lrate = 0.01
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    lr = LearningRateScheduler(step_decay)
    CB = [tb, cp, lr]
    input_shape = [x_train.shape[1], x_train.shape[2], x_train.shape[3]]
    print(x_train.shape)
    model = resnet_model(out_class=n_class, input_shape = input_shape)

    plot_model(model, show_layer_names=1)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=0.3,
              callbacks=CB, shuffle=1)

    loss, acc = model.evaluate(x_test, y_test, batch_size= BATCH_SIZE)
