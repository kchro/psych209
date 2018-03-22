# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from util import *

def baseline_japanese_classifier():
    X, y, le = load_japanese_dataset()
    print le.inverse_transform(y)

def train(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def plot_bar_graph(outputs, color):
    # count each
    counts = np.sum(outputs, axis=0)
    print counts
    plt.bar(range(15), counts, color=color)

if __name__ == '__main__':
    # 1/2/3): load japanese dataset
    # X, y = load_japanese_dataset()
    # X, y = load_english_dataset()
    # X, y = load_japanese_dataset()
    X, y = load_dataset()

    # 4) split into training and test sets
    # for svc:
    _, y  = decode_outputs(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 5) train the Random Forest
    print 'training the RF...'
    # rf = RandomForestClassifier(n_estimators=100)
    rf = RandomForestRegressor(n_estimators=100)
    # rf = SVR()
    rf.fit(X_train, y_train)

    EX, Ey = load_english_dataset()
    EX_train, EX_test, Ey_train, Ey_test = train_test_split(EX, Ey, test_size=0.3)
    #
    # test = np.vstack([X_test, EX_test])
    # y_test = np.vstack([y_test, Ey_test])

    print 'predicting y...'
    y_pred = rf.predict(X_test)
    # y_pred = rf.predict(EX_test)

    # y_pred = encode_labels(y_pred)
    # y_test = encode_labels(y_test)
    # plot_stacked_bars(y_pred, Ey_test, title="Random Forest Regressor (Bilingual)")
    # plt.show()
    # raise

    # plot_stacked_bars(y_pred, y_test, title="Random Forest Regressor (Bilingual)")
    # plt.show()

    print 'evaluating...'
    accuracy = accuracy_score(y_test, y_pred)
    print accuracy
