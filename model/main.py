import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras import backend as K
import matplotlib.pylab as plt

import data.load as load

K.set_image_dim_ordering('tf')

binDim = 128*2
batch_size = 10
nClasses = 6

(samples_train, samples_test), (x_train, y_train), (x_test, y_test) = load.getDataset(binDim)


x_train = x_train.reshape(samples_train, binDim, 14, 1)
x_test = x_test.reshape(samples_test, binDim, 14, 1)

#x_train = x_train.reshape(samples_train, batchDim * 14)
#x_test = x_test.reshape(samples_test, batchDim, 14, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

xmax = max(x_train.max(), x_test.max())

x_test /= xmax
x_train /= xmax

y_train = keras.utils.to_categorical(y_train, num_classes=nClasses)
y_test = keras.utils.to_categorical(y_test, num_classes=nClasses)

# normalize, initialize

epochs = 200

input_shape = (binDim, 14, 1,)

model = Sequential()
model.add(Conv2D(16, kernel_size=(20, 14), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape,
                 kernel_initializer='random_uniform'))
model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(42, kernel_size=(4, 8), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(600, activation='relu', kernel_initializer='random_uniform'))
model.add(Dense(60, activation='relu'))
model.add(Dense(nClasses, activation='softmax', kernel_initializer='random_uniform', use_bias = True, bias_initializer='random_uniform'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.001),
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






