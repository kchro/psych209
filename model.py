from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import Callback

import numpy as np
np.set_printoptions(precision=3, linewidth=np.nan)
from scipy.ndimage.filters import gaussian_filter1d

from util import decode_outputs

def get_model(input_dim, output_dim,
              num_layers=2, num_dim=256):

    model = Sequential()
    input_shape=(input_dim,)

    for i in range(num_layers):
        model.add(Dense(num_dim, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        input_shape = (num_dim,)

    model.add(Dense(output_dim))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adam')

    return model

class Checkpoint(Callback):
    def __init__(self, model, test_X, test_y):
        self.model = model
        self.test_X = test_X
        self.test_y = test_y

        _, self.true_labels = decode_outputs(test_y)

        self.activations = {}
        for label in self.true_labels:
            self.activations[label] = []

        self.pred_y = []

    def dump(self):
        data = []
        index = []
        for label in self.activations:
            epoch_outputs = np.vstack(self.activations[label])
            index.append(label)
            data.append(epoch_outputs.T)
        data = np.array(data)
        return index, data

    def on_epoch_end(self, epoch, logs={}):
        pred_y = self.model.predict(self.test_X)

        buckets = {}
        for i, output in enumerate(pred_y):
            label = self.true_labels[i]
            if label not in buckets:
                buckets[label] = []
            buckets[label].append(output)

        for label in self.activations:
            outputs = np.vstack(buckets[label])
            avg_output = np.mean(outputs, axis=0)
            self.activations[label].append(avg_output)

    def get_activations(self):
        data = {}
        for label in self.true_labels:
            epoch_outputs = np.vstack(self.activations[label])
            data[label] = epoch_outputs.T
        return data

def train(X, y, test_X=None, test_y=None,
          num_epochs=1, num_iterations=1, num_layers=2, num_dim=256):
    accs = []
    losses = []
    models = []
    ckpts = []
    best_model = None
    best_loss = float('inf')
    best_iter = None
    best_ckpt = None

    input_dim = len(X[0])
    output_dim = len(y[0])

    for iteration in range(num_iterations):
        print 'iter %d' % iteration
        model = get_model(input_dim, output_dim,
                          num_layers=num_layers,
                          num_dim=num_dim)
        ckpt = Checkpoint(model, test_X, test_y)
        data = model.fit(X, y,
                         batch_size=32,
                         epochs=num_epochs,
                         validation_split=0.15,
                         callbacks=[ckpt],
                         verbose=1)

        avg_loss = np.mean(data.history['loss'])
        if avg_loss < best_loss:
            # print 'updating best model'
            best_loss = avg_loss
            best_model = model
            best_iter = iteration
            best_ckpt = ckpt

        accs.append(data.history['acc'])
        losses.append(data.history['loss'])
        models.append(model)
        ckpts.append(ckpt)

    return best_model, best_iter, (best_ckpt, accs, losses, models, ckpts)

def retrain(models, X, y, test_X=None, test_y=None,
            num_epochs=1, num_iterations=1):
    accs = []
    losses = []
    _models = []
    ckpts = []
    best_model = None
    best_loss = float('inf')
    best_iter = None
    best_ckpt = None

    input_dim = len(X[0])
    output_dim = len(y[0])

    for iteration in range(num_iterations):
        print 'iter %d' % iteration
        # model = get_model(input_dim, output_dim)
        new_model = clone_model(models[iteration])
        new_model.set_weights(models[iteration].get_weights())
        new_model.compile(loss='categorical_crossentropy',
                          metrics=['accuracy'],
                          optimizer='adam')

        ckpt = Checkpoint(new_model, test_X, test_y)
        data = new_model.fit(X, y,
                             batch_size=32,
                             epochs=num_epochs,
                             validation_split=0.15,
                             callbacks=[ckpt],
                             verbose=1)

        avg_loss = np.mean(data.history['loss'])
        if avg_loss < best_loss:
            # print 'updating best model'
            best_loss = avg_loss
            best_model = new_model
            best_iter = iteration
            best_ckpt = ckpt

        accs.append(data.history['acc'])
        losses.append(data.history['loss'])
        _models.append(new_model)
        ckpts.append(ckpt)

    return best_model, best_iter, (best_ckpt, accs, losses, _models, ckpts)
