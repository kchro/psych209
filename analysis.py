# -*- coding: utf-8 -*-
from keras.models import load_model
import numpy as np
import json
import argparse

from plot_util import plot_acc_loss, plot_activations
from util import *

def get_parser():
    '''
    Set up argument parser
    Returns:
        parser: (ArgumentParser) the created parser
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--listener',
                        required=True,
                        choices={'english', 'EN',
                                 'japanese', 'JN',
                                 'j_learner', 'JL',
                                 'bilingual', 'BL'})
    parser.add_argument('--num_layers',
                        required=True,
                        type=int)
    parser.add_argument('--num_dim',
                        required=True,
                        type=int)
    return parser

class Checkpoint():
    def __init__(self, labels, data):
        self.true_labels = labels
        self.activations = {}
        for label, outputs in zip(labels, data):
            self.activations[label] = outputs

    def get_activations(self):
        return self.activations

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    listener = get_listener(args.listener)

    if listener == JAPANESE:
        spk = 'jn'
    elif listener == ENGLISH:
        spk = 'en'
    elif listener == JLEARNER:
        spk = 'jl2'
    else:
        spk = 'bl'

    lay = args.num_layers
    dim = args.num_dim

    model = load_model('weights/%s_model_l%d_d%d.h5' % (spk, lay, dim))
    accs = np.load('weights/%s_model_l%d_d%d_accs.npy' % (spk, lay, dim))
    losses = np.load('weights/%s_model_l%d_d%d_losses.npy' % (spk, lay, dim))

    with open('weights/%s_model_l%d_d%d.json' % (spk, lay, dim)) as json_data:
        obj = json.load(json_data)
        labels = obj['true_labels']
        i = obj['best_iteration']

    data = np.load('weights/%s_model_l%d_d%d_ckpt.npy' % (spk, lay, dim))

    ckpt = Checkpoint(labels, data)

    print spk, lay, dim
    try:
        print 'plotting...'
        labels = [u'ら', u'り', u'る', u'れ', u'ろ']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="%s_jp_act_l%d_d%d.png" % (spk, lay, dim))
    except KeyError:
        print 'no [r]'

    try:
        print 'plotting...'
        labels = [u'la', u'li', u'lu', u'le', u'lo']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="%s_el_act_l%d_d%d.png" % (spk, lay, dim))
    except KeyError:
        print 'no /l/'

    try:
        print 'plotting...'
        labels = [u'ra', u'ri', u'ru', u're', u'ro']
        plot_activations(ckpt, labels,
                         listener=listener,
                         filename="%s_er_act_l%d_d%d.png" % (spk, lay, dim))
    except KeyError:
        print 'no /r/'
    # plot_acc_loss(accs, losses, i, listener=1, filename='test')
    # labels = [u'ら', u'り', u'る', u'れ', u'ろ']
    # plot_activations(ckpt, labels, listener=1, filename='test')

    # print spk, lay, dim
    # # avg loss across the first N epochs
    # print 'avg loss 40', np.mean(np.mean(losses[:,:40], axis=0))
    # # avg final loss
    # print 'avg final', np.mean(losses[:,-1])
    #
    # print losses.shape
