# -*- coding: utf-8 -*-
import numpy as np
import pyaudio
from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa
from util import *

SIL_TH = -40

def match_target_amplitude(sound, target_dBFS):
    """
    apply gain to get audio at target dBFS
    """
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def make_chunks(audio_segment, chunk_length):
    """
    Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
    long.
    if chunk_length is 50 then you'll get a list of 50 millisecond long audio
    segments back (except the last one, which can be shorter)
    """
    number_of_chunks = np.ceil(len(audio_segment) / float(chunk_length))
    return [audio_segment[i * chunk_length:(i+1) * chunk_length]
            for i in range(int(number_of_chunks))]

def get_array(audio):
    """
    convert AudioSegment into np.array

    return:
        np.array, sample_rate
    """

    return np.array(audio.get_array_of_samples(), dtype=np.int32), audio.frame_rate

def get_chunks(filename):
    audio = AudioSegment.from_mp3(filename)

if __name__ == '__main__':
    test_file = AudioSegment.from_mp3("data/test_ら.wav")

    y, sr = get_array(test_file)
    print len(y), sr
    raise

    print 'average dBFS:', test_file.dBFS
    print 'silence thresh:', SIL_TH

    assert(test_file.dBFS > SIL_TH)

    chunks = split_on_silence(test_file,
                min_silence_len=300,
                silence_thresh=SIL_TH)

    target_dBFS = -16

    print 'number of chunks:', len(chunks)

    for i, chunk in enumerate(chunks):
        # chunk = match_target_amplitude(chunk, target_dBFS)
        y, sr = get_array(chunk)
        play(y, sr)

        print 'average dBFS per chunk %d: %f' % (i, chunk.dBFS)
        print 'length of chunk %d: %d' % (i, len(chunk))

        pause()

    print len(chunks)
