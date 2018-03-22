# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import accuracy_score

from keras.utils import np_utils
from keras.utils import plot_model

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from util import *
from parse import parse
from model import get_model
import argparse

def get_parser():
    '''
    Set up argument parser
    Returns:
        parser: (ArgumentParser) the created parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        action='store_true')
    parser.add_argument('--eval_only',
                        action='store_true')
    parser.add_argument('--listener',
                        required=True,
                        choices={'english', 'EN',
                                 'japanese', 'JN',
                                 'j_learner', 'JL',
                                 'bilingual', 'BL'})
    return parser

def plot_mfccs(letters, color):
    for i, letter in enumerate(letters):
        mfccs = np.load('data/npy/%s.npy' % letter)
        med_mfcc = np.median(mfccs[:], axis=0)[2:]
        med_mfcc /= np.linalg.norm(med_mfcc)
        plt.plot(med_mfcc, c=color)

def get_vowel_groups(english_l, english_r, japanese):
    groupings = []
    for e_l, e_r, jp in zip(english_l, english_r, japanese):
        groupings.append((e_l, e_r, jp))
    return groupings

def plot_acc_loss(accs, losses, listener):
    plt.figure(figsize=(7,6))

    title = 'Accuracy and Loss (%s)' % get_listener(listener)
    plt.suptitle(title)

    plt.subplot(211)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')

    for accuracy in accs:
        plt.plot(np.arange(len(accuracy))+1, accuracy)

    plt.subplot(212)
    plt.title('Model Loss')
    for loss in losses:
        plt.plot(np.arange(len(loss))+1, loss)

    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def plot_outputs(y_pred, y_test, listener):
    title = "Neural Network (%s)" % get_listener(listener)
    plot_stacked_bars(y_pred, y_test,
                      title=title,
                      large=False)

def train(X_train_dev, y_train_dev, num_iterations=1):
    accs = []
    losses = []

    best_model = None
    best_acc = -float('inf')
    best_grad = float('inf')
    best_iteration = None

    input_dim = len(X_train_dev[0])
    output_dim = len(label_list)

    for iteration in range(num_iterations):
        print 'iteration %d' % iteration
        model = get_model(input_dim, output_dim)
        history = model.fit(X_train_dev,
                            y_train_dev,
                            batch_size=32,
                            epochs=10,
                            validation_split=0.15,
                            verbose=0)

        loss_cap = 10
        loss = np.array(history.history['loss'][:loss_cap])
        avg_grad = np.mean(loss[1:] - loss[:-1])
        print loss
        print avg_grad

        final_acc = history.history['acc'][-1]
        loss_curve = history.history['loss']
        if avg_grad > best_grad:
            print 'updating best model'
            # best_acc = final_acc
            best_grad = avg_grad
            best_model = model
            best_iteration = iteration

        accs.append(history.history['acc'])
        losses.append(history.history['loss'])

    return best_model, best_iteration, accs, losses

if __name__ == '__main__':
    # parse arguments
    parser = get_parser()
    args = parser.parse_args()
    listener = get_listener(args.listener)

    # choose dataset
    if listener == JAPANESE or listener == JLEARNER:
        X, y = load_japanese_dataset()
    elif listener == ENGLISH:
        X, y = load_english_dataset()
    elif listener == BILINGUAL:
        X, y = load_dataset()

    # split the dataset
    X_td, X_test, y_td, y_test = split(X, y,
                                       test_size=0.2)

    # train model on train and dev sets
    model, iteration, accs, losses = train(X_td, y_td,
                                           num_iterations=10)

    plot_acc_loss(accs, losses, listener)
    y_pred = model.predict(X_test)
    plot_outputs(y_pred, y_test, listener)
    best_acc = accs[iteration]
    best_loss = losses[iteration]

    if listener == JLEARNER:
        # load english dataset
        X, y = load_dataset()
        X_td, X_test, y_td, y_test = split(X, y,
                                           test_size=0.2)

        # retrain model
        history = model.fit(X_td, y_td,
                            batch_size=32,
                            epochs=30,
                            validation_split=0.15)

        accs = [best_acc, history.history['acc']]
        losses = [best_loss, history.history['loss']]
        plot_acc_loss(accs, losses, listener)
        y_pred = model.predict(X_test)
        plot_outputs(y_pred, y_test, listener)
