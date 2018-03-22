# -*- coding: utf-8 -*-
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.font_manager
fontpath = '/Library/Fonts/Osaka.ttf'
properties = matplotlib.font_manager.FontProperties(fname=fontpath)
matplotlib.rcParams['font.family'] = properties.get_name()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
import librosa
from parse import parse
from util import *

colors = ['#E57373', '#F44336', '#D32F2F',
          '#BA68C8', '#9C27B0', '#7B1FA2',
          '#7986CB', '#3F51B5', '#303F9F',
          '#81C784', '#4CAF50', '#388E3C',
          '#FFB74D', '#FF9800', '#F57C00']

def plot_mfcc():
    dirname = 'data'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    paths = []
    def init():
        for letter, color in zip(label_list, colors):
            # skip over metadata in directory
            filename = u'data/npy/%s.npy' % letter
            mfccs = np.load(filename)
            scat = ax.scatter(mfccs.T[0], mfccs.T[1], mfccs.T[2], color=color, label=letter)
            paths.append(scat)
        return paths

    # legend = plt.legend()
    def animate(i):
        ax.view_init(elev=10., azim=i)
        plt.legend()
        return tuple(paths)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)

    anim.save('MFCC0.gif', fps=30, writer='imagemagick')

    raise

def plot_acc_loss(accs, losses, best_iter, listener=0, filename=''):
    plt.figure(figsize=(7,6))

    title = 'Accuracy and Loss (%s)' % get_listener(listener)
    plt.suptitle(title)

    plt.subplot(211)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')

    for i, accuracy in enumerate(accs):
        plt.plot(np.arange(len(accuracy))+1, accuracy)
        if i == best_iter:
            plt.plot(np.arange(len(accuracy))+1, accuracy, 'bo')

    plt.subplot(212)
    plt.title('Model Loss')
    for i, loss in enumerate(losses):
        plt.plot(np.arange(len(loss))+1, loss)
        if i == best_iter:
            plt.plot(np.arange(len(loss))+1, loss, 'ro')

    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(filename)
    # plt.show()

def plot_outputs(outputs, truths, listener=0, filename='', large=True):
    """
    plot stacked bars, where the x-axis represents the
    activated letter and the colored bars represent
    which was the true letter presented.
    """
    pred_codes, pred_labels = decode_outputs(outputs)
    true_codes, true_labels = decode_outputs(truths)

    l_outputs = []
    r_outputs = []
    j_outputs = []
    for i, true in enumerate(true_codes):
        if true % 3 == 0:
            j_outputs.append(outputs[i])
        if true % 3 == 1:
            l_outputs.append(outputs[i])
        if true % 3 == 2:
            r_outputs.append(outputs[i])

    j_counts = np.sum(j_outputs, axis=0)
    l_counts = np.sum(l_outputs, axis=0)
    r_counts = np.sum(r_outputs, axis=0)

    print j_counts
    print l_counts
    print r_counts

    if not isinstance(j_counts, np.ndarray):
        j_counts = np.zeros(15)
    if not isinstance(l_counts, np.ndarray):
        l_counts = np.zeros(15)
    if not isinstance(r_counts, np.ndarray):
        r_counts = np.zeros(15)

    totals = j_counts + l_counts + r_counts
    # avoid divide by 0 errors
    for i in range(len(totals)):
        if np.isclose(totals[i], 0):
            totals[i] = 1.0

    # normalize to a percentage
    # j_counts /= totals
    # l_counts /= totals
    # r_counts /= totals

    width = 0.3
    pos = np.array([1, 2, 3,
                    5, 6, 7,
                    9, 10, 11,
                    13, 14, 15,
                    17, 18, 19]) * width
    xticks = pos + width/2.

    if large:
        figsize=(20,10)
    else:
        figsize=(10,5)

    plt.figure(figsize=figsize)
    plt.title('Activations (%s)' % get_listener(listener))

    j_plot = plt.bar(pos, j_counts, width,
                     label=u'Japanese')
    l_plot = plt.bar(pos, l_counts, width,
                     label=u'English L',
                     bottom=j_counts)
    r_plot = plt.bar(pos, r_counts, width,
                     label=u'English R',
                     bottom=j_counts+l_counts)

    print j_plot

    plt.xticks(xticks, label_list)
    # plt.yticks(np.arange(0,1.1,0.1))
    plt.xlabel('letter activations')
    # plt.ylabel('percentage')
    plt.legend(loc='center left',
               bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)
    plt.show()

def plot_activations(ckpt, desired_labels, cap=1000, listener=0, filename=''):
    activations = ckpt.get_activations()

    plt.figure(figsize=(5, 9))
    for i, label in enumerate(desired_labels):
        ax = plt.subplot('%d1%d' % (len(desired_labels), i+1))
        ax.set_ylabel(u'%s' % label, rotation='horizontal', labelpad=10)
        for j, y in enumerate(activations[label]):
            y = y[:cap]
            x = np.arange(len(y), dtype=int) + 1
            if j % 3 == 0:
                style = '-'
            elif j % 3 == 1:
                style = ':'
            else:
                style = '-.'

            plt.plot(x, y, c=colors[j], label=label_list[j], linestyle=style)
        if i < len(desired_labels) - 1:
            plt.tick_params(labelleft=False, labelbottom=False)
        else:
            plt.tick_params(labelleft=False)

    plt.suptitle('Activations (%s)' % get_listener(listener))
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, left=0.05, right=0.85, top=0.95, bottom=0.05)
    plt.legend(loc='center left', bbox_to_anchor=(1, 3))
    plt.savefig(filename)
    # plt.show()

if __name__ == '__main__':
    plot_mfcc()
