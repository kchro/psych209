# -*- coding: utf-8 -*-
import os
from sklearn.model_selection import train_test_split as split
from util import encode_labels
import numpy as np

ENGLISH, JAPANESE, JLEARNER, BILINGUAL = range(4)

def load_all_datasets():
    """
    load a dataset of audio
    """
    datasets = {}

    mfccs = []
    labels = []
    for filename in os.listdir('data/npy'):
        # load 100 input vectors
        mfcc = np.load('data/npy/%s'%filename)
        letter = filename[:-4]
        labels = encode_labels([letter]*len(mfcc))
        datasets[filename[:-4].decode('utf-8')] = (mfcc, labels)

    return datasets

def pstack(arr):
    """
    vstack and permute by row
    """
    return np.random.permutation(np.vstack(arr))

def load_balanced_datasets():
    print 'loading...'
    datasets = load_all_datasets()
    print

    J_train_X = []
    J_train_y = []
    J_test_X = []
    J_test_y = []
    E_train_X = []
    E_train_y = []
    E_test_X = []
    E_test_y = []

    for letter in datasets:
        # split by individual letter,
        # distribute to train and set sets
        X, y = datasets[letter]
        train_X, test_X, train_y, test_y = split(X, y, test_size=0.2)
        if letter in u'らりるれろ':
            J_train_X.append(train_X)
            J_train_y.append(train_y)
            J_test_X.append(test_X)
            J_test_y.append(test_y)
        else:
            E_train_X.append(train_X)
            E_train_y.append(train_y)
            E_test_X.append(test_X)
            E_test_y.append(test_y)

    # randomize order of data
    J_train_X, _, J_train_y, _ = split(np.vstack(J_train_X),
                                       np.vstack(J_train_y),
                                       test_size=0)
    J_test_X, _, J_test_y, _ = split(np.vstack(J_test_X),
                                     np.vstack(J_test_y),
                                     test_size=0)
    E_train_X, _, E_train_y, _ = split(np.vstack(E_train_X),
                                       np.vstack(E_train_y),
                                       test_size=0)
    E_test_X, _, E_test_y, _ = split(np.vstack(E_test_X),
                                     np.vstack(E_test_y),
                                     test_size=0)

    # package data
    J_data = (J_train_X, J_test_X, J_train_y, J_test_y)
    E_data = (E_train_X, E_test_X, E_train_y, E_test_y)
    return J_data, E_data

def interpolate(X1, X2, y1, y2, inter=0.5):
    X1_cutoff = int(len(X1) * inter)
    X2_cutoff = len(X1) - X1_cutoff
    X = np.vstack([X1[:X1_cutoff], X2[:X2_cutoff]])
    y = np.vstack([y1[:X1_cutoff], y2[:X2_cutoff]])
    X, _, y, _ = split(X, y, test_size=0)
    return X, y

def save_datasets():
    J_data, E_data = load_balanced_datasets()
    J_train_X, J_test_X, J_train_y, J_test_y = J_data
    E_train_X, E_test_X, E_train_y, E_test_y = E_data
    np.save('j_train_x.npy', J_train_X)
    np.save('j_test_x.npy', J_test_X)
    np.save('j_train_y.npy', J_train_y)
    np.save('j_test_y.npy', J_test_y)
    np.save('e_train_x.npy', E_train_X)
    np.save('e_test_x.npy', E_test_X)
    np.save('e_train_y.npy', E_train_y)
    np.save('e_test_y.npy', E_test_y)

def load_dataset(language, inter=1):
    J_train_X = np.load('j_train_x.npy')
    J_test_X = np.load('j_test_x.npy')
    J_train_y = np.load('j_train_y.npy')
    J_test_y = np.load('j_test_y.npy')
    E_train_X = np.load('e_train_x.npy')
    E_test_X = np.load('e_test_x.npy')
    E_train_y = np.load('e_train_y.npy')
    E_test_y = np.load('e_test_y.npy')

    if language == JAPANESE:
        J_data = (J_train_X, J_test_X, J_train_y, J_test_y)
        E_data = (E_test_X, E_test_y)
    elif language == ENGLISH:
        J_data = None
        E_data = (E_train_X, E_test_X, E_train_y, E_test_y)
    elif language == JLEARNER:
        J_data = (J_train_X, J_test_X, J_train_y, J_test_y)
        E_data = (E_train_X, E_test_X, E_train_y, E_test_y)
    elif language == BILINGUAL:
        J_data = (J_train_X, J_test_X, J_train_y, J_test_y)
        E_data = (E_train_X, E_test_X, E_train_y, E_test_y)
    else:
        train_X, train_y = interpolate(J_train_X, E_train_X, J_train_y, E_train_y, inter=inter)
        test_X, test_y = interpolate(J_test_X, E_test_X, J_test_y, E_test_y, inter=inter)
        J_data = (train_X, test_X, train_y, test_y)
        E_data = (E_train_X, E_test_X, E_train_y, E_test_y)

    return J_data, E_data
