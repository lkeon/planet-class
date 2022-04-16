import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import backend
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import gradient_descent_v2
 

# load train and test dataset
def load_numpy_dataset(file):
    # load dataset
    data = np.load(file)
    X, y = data['arr_0'], data['arr_1']
    # separate into train and test datasets
    trainX, testX, trainY, testY = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=1)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return trainX, trainY, testX, testY
 

# calculate fbeta score for multi-class/label classification
def fbeta(ytrue, ypred, beta=2):
    # clip predictions
    ypred = backend.clip(ypred, 0, 1)
    # calculate elements
    tp = backend.sum(backend.round(backend.clip(ytrue * ypred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(ypred - ytrue, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(ytrue - ypred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + backend.epsilon())
    # calculate recall
    r = tp / (tp + fn + backend.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * p*r / (bb*p + r + backend.epsilon()))
    return fbeta_score

 
# define cnn model
def define_model(in_shape=(128, 128, 3), out_shape=17):
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same',
                     input_shape=in_shape))
    model.add(Conv2D(32, (3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(Conv2D(64, (3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(Conv2D(128, (3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,
                    activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(out_shape, activation='sigmoid'))
    # compile model
    opt = gradient_descent_v2.SGD(lr=0.01,
                                  momentum=0.9)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[fbeta])
    return model

 
# plot diagnostic learning curves
def print_history(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Fbeta')
    plt.plot(history.history['fbeta'], color='blue', label='train')
    plt.plot(history.history['val_fbeta'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()

 
# run classification to train the model
def run_classification():
    # load dataset
    file = 'data/planet_data.npz'
    trainX, trainY, testX, testY = load_numpy_dataset(file)
    # create data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    # prepare iterators
    train_it = datagen.flow(trainX, trainY, batch_size=128)
    test_it = datagen.flow(testX, testY, batch_size=128)
    # define model
    model = define_model()
    # fit model
    history = model.fit(train_it,
                        steps_per_epoch=len(train_it),
                        validation_data=test_it,
                        validation_steps=len(test_it),
                        epochs=200,
                        verbose=1)
    # evaluate model
    loss, fbeta = model.evaluate(test_it,
                                 steps=len(test_it),
                                 verbose=0)
    print('Loss: {}, fbeta: {}'.format(loss, fbeta))
    # learning curves
    print_history(history)
 

# entry point, run the test harness
if __name__ == '__main__':
    run_classification()
