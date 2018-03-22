# -*- coding: utf-8 -*-
import os
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
# fontpath = '/Library/Fonts/Osaka.ttf'
# properties = matplotlib.font_manager.FontProperties(fname=fontpath)
# matplotlib.rcParams['font.family'] = properties.get_name()

import matplotlib.pyplot as plt
import librosa
import librosa.display
from time import sleep
from sklearn.preprocessing import LabelEncoder
# from parse import parse

# english_l = u'la li lu le lo'.split()
# english_r = u'ra ri ru re ro'.split()
# japanese_kana = u'ら り る れ ろ'.split()
# label_list = english_l + english_r + japanese_kana

label_list = u'ら la ra り li ri る lu ru れ le re ろ lo ro'.split()

ENGLISH, JAPANESE, JLEARNER, BILINGUAL = range(4)

def get_listener(listener):
    """
    return appropriate code from listener
    """
    if listener == 'english' or listener == 'EN':
        return ENGLISH
    if listener == 'japanese' or listener == 'JN':
        return JAPANESE
    if listener == 'j_learner' or listener == 'JL':
        return JLEARNER
    if listener == 'bilingual' or listener == 'BL':
        return BILINGUAL
    if isinstance(listener, int):
        tags = ['English', 'Japanese', 'Learner', 'Bilingual']
        return tags[listener]

def pause():
    # simple pause for processing / debugging
    sleep(0.200)

def play(y, sr):
    # simple play for processing / debugging
    sd.play(y, sr)
    sd.wait()

def get_label_code(label):
    """
    from a given label, return the encoded index
    """
    return label_list.index(label.decode('utf-8'))

def encode_labels(labels, num_labels=15):
    """
    from a list of labels, generate a list of one-hot
    encodings
    """
    outputs = []
    for label in labels:
        zero_vector = np.zeros(num_labels)
        index = get_label_code(label)
        zero_vector[index] = 1.0
        outputs.append(zero_vector)
    outputs = np.array(outputs)
    return outputs

def decode_outputs(outputs):
    """
    from a list of one-hot vectors, generate a list of
    indices and associated letters
    """
    _, codes = np.where(outputs==1.0)
    labels = []
    for code in codes:
        label = label_list[code]
        labels.append(label)
    return codes, labels

def plot_waveform(letter):
    chunks, sr, mfccs = parse('data', '%s.wav' % letter, save=False)

    plt.figure(figsize=(10, 6))
    plt.subplot(311)
    plt.plot(chunks[0])
    plt.title(u'waveform of /%s/' % letter)
    plt.xlabel('samples')
    plt.ylabel('amplitude')
    plt.subplot(312)
    plt.title('Mel-Frequency Cepstrum')
    S = librosa.feature.melspectrogram(chunks[0], sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.subplot(313)
    plt.title('Mel-Frequency Cepstrum Coefficients')
    plt.plot(range(1, len(mfccs[0])+1), mfccs[0], c='orange')
    plt.xticks(range(1, len(mfccs[0])+1))
    plt.xlabel('coefficient')
    plt.ylabel('magnitude')
    plt.tight_layout()

    plt.savefig(u'data/plots/%s' % letter)
    plt.close()

def plot_stacked_bars0(outputs, truths, title='', large=True):
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
    plt.title(title)

    j_plot = plt.bar(pos, j_counts, width,
                     color='#F44336',
                     label=u'Japanese')
    l_plot = plt.bar(pos, l_counts, width,
                     color='#2196F3',
                     label=u'English L',
                     bottom=j_counts)
    r_plot = plt.bar(pos, r_counts, width,
                     color='#9C27B0',
                     label=u'English R',
                     bottom=j_counts+l_counts)

    plt.xticks(xticks, label_list)
    # plt.yticks(np.arange(0,1.1,0.1))
    plt.xlabel('letter activations')
    # plt.ylabel('percentage')
    plt.legend(loc='center left',
               bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)
    plt.show()
