# -*- coding: utf-8 -*-
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from util import *

def mel_power_spec(y, sr):
    """
    plot mel power spectrogram
    """
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12,4))
    librosa.display.specshow(log_S,
                             sr=sr,
                             x_axis='time',
                             y_axis='mel')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()

def extract_features(y, sr):
    # extract mfcc (ignore gain)
    mfcc = np.mean(librosa.feature.mfcc(y, sr).T, axis=0)
    return mfcc

def split_on_silence(y, sr):
    """
    split the audio file by silence

    silence is defined as top_db dB below
            20 * np.log10( np.max(y) )
    """
    intervals = librosa.effects.split(y, top_db=10)

    chunks = []
    for start, end in intervals:
        chunk = y[start:end]
        chunks.append(chunk)

    return chunks

def parse(dirname, filename, save=True):
    """
    1) load a file as wav
    2) split wav on silence
    3) check validity of each chunk
    4) calculate the mfcc averaged
       across time
    5) store all mfcc's as numpy array
    6) save numpy array
    """
    y, sr = librosa.load('%s/%s' %(dirname,filename), res_type='kaiser_fast')

    chunks = split_on_silence(y, sr)

    print 'found %d chunks' % len(chunks)

    features = []
    for chunk in chunks:
        # debugging:
        # check validity of chunk
        # play(chunk, sr)
        # mel_power_spec(chunk, sr)

        mfcc = extract_features(chunk, sr)
        features.append(mfcc)

    features = np.array(features)
    if save:
        np.save('%s/npy/%s.npy' % (dirname, filename[:-4]), features)

    return chunks, sr, features

if __name__ == '__main__':
    debug = False

    dirname = 'data'
    for filename in os.listdir(dirname):
        # skip over metadata in directory
        if '.wav' not in filename:
            continue
        if not debug and 'test' in filename:
            continue

        print 'parsing', filename
        parse(dirname, filename)
