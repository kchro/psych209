# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import argparse

def get_parser():
    '''
    Set up argument parser
    Returns:
        parser: (ArgumentParser) the created parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--listener', choices={'english', 'japanese', 'j_learner', 'bilingual'})
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print args.listener

    raise

    # objective of unit test:
    # take two sound samples 'ga' and 'ro'
    # see if we can train a model to separate the two sounds

    # 1) load the sounds as npy
    ga_mfcc = np.load('data/npy/ら.npy')
    ro_mfcc = np.load('data/npy/ro.npy')

    ga_mfcc /= np.linalg.norm(ga_mfcc)
    ro_mfcc /= np.linalg.norm(ro_mfcc)

    # 1.5) preprocess sounds?
    # ga_mfcc = ga_mfcc / np.linalg.norm(ga_mfcc)
    # ro_mfcc = ro_mfcc / np.linalg.norm(ro_mfcc)

    # 2) create labels
    ga_label = np.zeros(ga_mfcc.shape[0])
    ro_label = np.ones(ro_mfcc.shape[0])

    # 3) combine inputs and outputs
    X = np.vstack([ga_mfcc, ro_mfcc])
    y = np.hstack([ga_label, ro_label])

    # 4) split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 5) train the SVM
    print 'training the SVM...'
    svm = SVC()
    svm.fit(X_train, y_train)

    print 'predicting y...'
    y_pred = svm.predict(X_test)

    print 'evaluating...'
    for pred, test in zip(y_pred, y_test):
        print pred, test
