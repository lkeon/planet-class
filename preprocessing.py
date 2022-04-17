import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from pandas import read_csv
from os import listdir
from keras.preprocessing.image import load_img, img_to_array


def plot_train_images(folder):
    ''' Plot 9 random images in the dataset.
    '''
    imageRand = np.random.randint(0, high=40479, size=9)

    for iteration, integer in enumerate(imageRand):
        plt.subplot(3, 3, iteration+1)
        filename = 'train_' + str(integer) + '.jpg'
        path = folder + filename
        image = imread(path)
        plt.imshow(image)

    plt.show()


def get_labels_maps(fileName):
    ''' Return dictionaries maping from labels to numbers
    and vice versa.
    '''
    tagsCsv = read_csv(fileName)
    # Get all unique tags appearing in the database
    labels = set()
    for i in range(len(tagsCsv)):
        tags = tagsCsv['tags'][i].split()
        labels.update(tags)

    # Convert set to list and sort
    labels = list(labels)
    labels.sort()

    # Create dictionaries converting labels to num
    labelToNum = {labels[i]: i for i in range(len(labels))}
    numToLabel = {i: labels[i] for i in range(len(labels))}

    return labelToNum, numToLabel


def get_train_labels(fileName):
    ''' Get a dictionary linking image names to tags.
    '''
    # Read CSV file
    tagsCsv = read_csv(fileName)
    trainLabels = dict()

    for i in range(len(tagsCsv)):
        name, tags = tagsCsv['image_name'][i], tagsCsv['tags'][i]
        trainLabels[name] = tags.split()

    return trainLabels


def one_hot_encode(tags, labelToNum):
    ''' Create one hot encoded vector for each image.
    '''
    trainHot = np.zeros(len(labelToNum), dtype='uint8')
    for tag in tags:
        trainHot[labelToNum[tag]] = 1
    return trainHot


def load_dataset(path, trainLabels, labelToNum, number=None, imageSize=128):
    ''' Import images and labels.
    '''
    photos, labels = list(), list()
    files = listdir(path)

    if number is not None:
        files = files[:number]

    for file in files:
        photo = load_img(path + file, target_size=(imageSize, imageSize))
        photo = img_to_array(photo, dtype='uint8')
        tags = trainLabels[file[:-4]]
        label = one_hot_encode(tags, labelToNum)
        photos.append(photo)
        labels.append(label)
    X = np.asarray(photos, dtype='uint8')
    y = np.asarray(labels, dtype='uint8')
    return X, y


if __name__ == '__main__':
    fileName = 'data/train_v2.csv'
    labelToNum, numToLabel = get_labels_maps(fileName)
    trainLabels = get_train_labels(fileName)
    X, y = load_dataset('data/train-jpg/', trainLabels, labelToNum)
    print(X.shape, y.shape)
    np.savez_compressed('data/planet_data.npz', X, y)
    print('Dataset saved to disk.')
