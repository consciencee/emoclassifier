import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GaussianNoise
from keras.models import Sequential
from keras import backend as K
from keras.datasets import mnist
import matplotlib.pylab as plt
import numpy as np

import sys
import pickle

import data.load as load

K.set_image_dim_ordering('tf')

binDim = 128*2
batch_size = 10
nClasses = 10

#import gzip
#f = gzip.open('mnist.pkl.gz', 'rb')
#if sys.version_info < (3,):
#    data = pickle.load(f)
#else:
#    data = pickle.load(f, encoding='bytes')
#f.close()

#(x_train,y_train), (x_test,y_test) = data

#x_train = x_train[0:1000]
#x_test = x_test[0:200]
#y_train = y_train[0:1000]
#y_test = y_test[0:200]

#x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

(samples_train, samples_test), (x_train, y_train), (x_test, y_test) = load.getDataset(binDim)

print samples_train, samples_test

x_train = x_train.reshape(samples_train, binDim, 14, 1)
x_test = x_test.reshape(samples_test, binDim, 14, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

xmax = max(x_train.max(), x_test.max())

x_test /= xmax
x_train /= xmax

x_train = x_train + x_train.copy()

y_train = keras.utils.to_categorical(y_train, num_classes=nClasses)
y_test = keras.utils.to_categorical(y_test, num_classes=nClasses)

y_train = y_train + y_train.copy()



# normalize, initialize

epochs = 200

input_shape = (binDim, 14, 1,)
#input_shape = (28, 28, 1,)


model = Sequential()
model.add(GaussianNoise(stddev=0.001, input_shape=input_shape))
model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                 kernel_initializer='he_normal'))
#model.add(Conv2D(42, kernel_size=(32, 2), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 2)))
model.add(Conv2D(42, kernel_size=(4, 8), padding='same', activation='relu', kernel_initializer='he_normal'))
#model.add(Conv2D(128, kernel_size=(8, 8), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
#model.add(Dense(2048, activation='relu'))
#model.add(Dense(2048, activation='relu'))
model.add(Dense(600, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(60, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(nClasses, activation='softmax', kernel_initializer='glorot_normal', use_bias = True))




model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.adam(lr=0.001),
              metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history]
          )

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.plot(range(epochs), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


#model.save('alexey2.h5')









