import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import backend
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import gradient_descent_v2
from keras.applications.vgg16 import VGG16
 

def load_numpy_dataset(file):
    ''' Load database that was saved as npz.
    '''
    data = np.load(file)
    X, y = data['arr_0'], data['arr_1']
    # separate into train and test datasets
    trainX, testX, trainY, testY = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=1)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return trainX, trainY, testX, testY
 

def fbeta(ytrue, ypred, beta=2):
    ''' Calculate fbeta score for multilabel classification.
    '''
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

 
def define_model(in_shape=(128, 128, 3), out_shape=17):
    ''' Define sequential model.
    '''
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


def define_vgg_model(in_shape=(128, 128, 3), out_shape=17, trainable=False):
    ''' This function returns a VGGG model as defined in Keras.
    '''
    # load model from Keras
    model = VGG16(include_top=False, input_shape=in_shape)
    # set layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # allow last block to be trinable
    if trainable:
        model.get_layer('block5_conv1').trainable = True
        model.get_layer('block5_conv2').trainable = True
        model.get_layer('block5_conv3').trainable = True
        model.get_layer('block5_pool').trainable = True
    # add new layers for classification
    flatten_layer = Flatten()(model.layers[-1].output)
    class_layer = Dense(128,
                        activation='relu',
                        kernel_initializer='he_uniform')(flatten_layer)
    output = Dense(out_shape, activation='sigmoid')(class_layer)
    # create new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = gradient_descent_v2.SGD(lr=0.01,
                                  momentum=0.9)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[fbeta])
    return model

 
def print_history(history):
    ''' Show plot history.
    '''
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.legend()
    # plot accuracy
    plt.subplot(212)
    plt.title('Fbeta')
    plt.plot(history.history['fbeta'], color='blue', label='train')
    plt.plot(history.history['val_fbeta'], color='orange', label='test')
    plt.legend()
    plt.show()

 
def run_classification(mode='default', trainable=False):
    ''' Main function to be run.
    '''
    # load dataset
    file = 'data/planet_data.npz'
    trainX, trainY, testX, testY = load_numpy_dataset(file)
    # create data generator
    if mode == 'default':
        model = define_model()
        datagen_train = ImageDataGenerator(rescale=1/255,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           rotation_range=90)
        datagen_test = ImageDataGenerator(rescale=1/255)
    elif mode == 'vgg':
        model = define_vgg_model(trainable=trainable)
        datagen_train = ImageDataGenerator(featurewise_center=True,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           rotation_range=90)
        datagen_test = ImageDataGenerator(featurewise_center=True)
    # specify imagenet mean values for centering
    datagen_train.mean = [123.68, 116.779, 103.939]
    datagen_test.mean = [123.68, 116.779, 103.939]
    # prepare iterators
    train_it = datagen_train.flow(trainX, trainY, batch_size=12)
    test_it = datagen_test.flow(testX, testY, batch_size=12)
    # fit model
    history = model.fit(train_it,
                        steps_per_epoch=len(train_it),
                        validation_data=test_it,
                        validation_steps=len(test_it),
                        epochs=10,
                        verbose=1)
    # evaluate model
    loss, fbeta = model.evaluate(test_it,
                                 steps=len(test_it),
                                 verbose=0)
    print('Loss: {}, fbeta: {}'.format(loss, fbeta))
    print_history(history)
 

# Run if executed as a script
if __name__ == '__main__':
    run_classification()
