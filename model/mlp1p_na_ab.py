import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GaussianNoise
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
import matplotlib.pylab as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib


import data.load as load

K.set_image_dim_ordering('tf')

binDim = 128*2
batch_size = 10
nClasses = 10
epochs = 200
input_shape = (binDim, 14, 1,)

def create_model():

    model = Sequential()
    model.add(GaussianNoise(stddev=0.001, input_shape=input_shape)) # TODO
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                 kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 2)))
    model.add(Conv2D(42, kernel_size=(4, 8), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
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

    checkpoint = ModelCheckpoint("mlp1p_na_ab_autosave.h5", monitor='val_acc', save_best_only=True)
    csvlogger = CSVLogger("mlp1p_na_ab.csv")

    classifier = KerasClassifier(build_fn=model, batch_size=batch_size, nb_epoch=epochs, callbacks=[history, checkpoint, csvlogger])

    return AdaBoostClassifier(base_estimator=classifier)

def prepare_data(filenames, labels, ignoredIDs = ()):

    (samples_train, samples_test), (x_train, y_train), (x_test, y_test) = load.getDataset(binDim, filenames, labels, ignoredIDs)

    x_train = x_train.reshape(samples_train, binDim, 14, 1)
    x_test = x_test.reshape(samples_test, binDim, 14, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, num_classes=nClasses)
    y_test = keras.utils.to_categorical(y_test, num_classes=nClasses)

    # normalize
    xmax = max(x_train.max(), x_test.max())
    x_test /= xmax
    x_train /= xmax


    # doubling - as there ll be noise augmentation
    y_train = y_train + y_train.copy()
    x_train = x_train + x_train.copy()

    return x_train, y_train, x_test, y_test


def run(model, x_train, y_train, x_test, y_test):


    model.fit(x_train, y_train)

    return model.score(x_train, y_train), model.score(x_test, y_test)



model = create_model()

filenames = ["../samples/Alexey/2/Alexey1_2_eeg_log.csv",
             "../samples/Alexey/2/Alexey2_2_eeg_log.csv",
             "../samples/Alexey/2/Alexey3_2_eeg_log.csv",
             "../samples/Alexey/2/Alexey4_2_eeg_log.csv",
             "../samples/Alexey/2/Alexey5_2_eeg_log.csv",
             "../samples/Alexey/2/Alexey6_2_eeg_log.csv"]
labels = [1, 2, 3, 4, 5, 0]
ignoredIDs = ("Session4", "Session1")
x_train, y_train, x_test, y_test = prepare_data(filenames, labels, ignoredIDs)
scoretrain, scoretest = run(model, x_train, y_train, x_test, y_test)
print('Train accuracy:', scoretrain)
print('Test accuracy:', scoretest)
#plt.plot(range(epochs), history_acc)
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.show()

joblib.dump(model, 'mlp1p_na_ab_end.joblib.pkl')
