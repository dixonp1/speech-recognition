import random
import math
import hashlib
import os.path

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile

'''
load data for training
get input from mic

class audio_processor
    processes audio
    pad input
    extract features

'''

SILENCE_PERCENTAGE = 10
SILENCE_INDEX = 0
UNKNOWN_PERCENTAGE = 10
UNKNOWN_INDEX = 1
RANDOM_SEED = 48237

PRE_EMPHASIS = 0.97
FRAME_SIZE = 0.025  # s
FRAME_STRIDE = 0.01  # s
MAX_SHIFT = 100  # ms
SAMPLE_RATE = 16000
MAX_SAMPLES = 16000

FFT_LEN = 512
NUM_MEL_FILTERBANKS = 40
LOW_FREQUENCY = 0  # hz
HIGH_FREQUENCY = SAMPLE_RATE / 2  # hz


# WORD_LIST =

class AudioProcessor:

    def __init__(self, training=False):
        self.shift_amt = 0
        self._calc_mel_filterbanks()
        self.create_processing_graph(training)

    def load_wav_file(self, filename):
        with tf.Session(graph=tf.Graph()) as sess:
            wav_filename = tf.placeholder(tf.string, [])
            loader = io_ops.read_file(wav_filename)
            decoder = audio_ops.decode_wav(loader, desired_channels=1)
            return sess.run(decoder, feed_dict={wav_filename: filename}).audio.flatten()

    def create_processing_graph(self, training=False):
        """
        :return:
        """
        '''
        load a file
        apply pre-emphasis
        randomly shift in time to apply distortion if training
        pad to ensure evenly windowed
        extract features
        return features

        '''
        self.filename_placeholder = tf.placeholder(tf.string)
        loader = io_ops.read_file(self.filename_placeholder)
        signal = audio_ops.decode_wav(loader,
                                      desired_channels=1,
                                      desired_samples=MAX_SAMPLES).audio

        self.volume_placeholder = tf.placeholder(tf.float32)
        signal = signal * self.volume_placeholder

        emphasized_signal = tf.subtract(signal[1:], PRE_EMPHASIS * signal[:-1])

        frame_length = tf.constant(round(FRAME_SIZE * SAMPLE_RATE), dtype=tf.int32)
        frame_step = tf.constant(round(FRAME_STRIDE * SAMPLE_RATE), dtype=tf.int32)
        signal_len = tf.shape(emphasized_signal)[0]
        diff = signal_len - frame_length
        num_frames = tf.cast(tf.ceil(diff / frame_step) + 1, tf.int32)

        # pad signal to ensure evenly broken into frames
        pad_amt = (num_frames - 1) * frame_step + frame_length - signal_len

        if training:
            self.shift_amt = tf.placeholder(tf.int32, [2])
            self.offset = tf.placeholder(tf.int32, [2])
            padded_signal = tf.pad(emphasized_signal, [self.shift_amt, [0, 0]], mode='CONSTANT')
            padded_signal = tf.slice(padded_signal, self.offset, [MAX_SAMPLES-1, -1])
        else:
            self.shift_amt = 0
            padded_signal = emphasized_signal

        padded_signal = tf.pad(padded_signal, [[0, pad_amt], [0, 0]], mode='CONSTANT')
        padded_signal = tf.reshape(padded_signal, [-1])

        self.padded_signal = padded_signal

        # split signal into frames
        t1 = tf.tile([tf.range(0, frame_length)], [num_frames, 1])
        t2 = tf.tile([tf.range(0, num_frames * frame_step, frame_step)], [frame_length, 1])
        indices = t1 + tf.transpose(t2)
        frames = tf.gather(padded_signal, indices)

        # apply hamming window
        hwin = tf.range(0, frame_length)
        theta = tf.cast(2 * hwin, dtype=tf.float32)
        frames *= 0.54 - 0.46 * tf.cos(math.pi * theta / (tf.cast(frame_length - 1, dtype=tf.float32)))

        # perform FFT and get power spectrum
        frame_fft_abs = tf.abs(tf.spectral.rfft(frames, fft_length=[FFT_LEN]))
        power_spectrum = frame_fft_abs * frame_fft_abs / FFT_LEN

        audio_filterbanks = tf.matmul(power_spectrum, tf.cast(self.mel_fbanks, dtype=tf.float32), transpose_b=True)
        # filterbanks = tf.log(filterbanks)
        audio_filterbanks -= tf.reduce_mean(audio_filterbanks)
        self.features = audio_filterbanks

    '''
    def get_data(self, path, sess, batch_size):

        folder = os.path.join(path, '*.wav')
        files = gfile.Glob(folder)[:50]

        data = np.empty((0, MAX_SIGNAL_LEN))
        for f in files:
            sig = sess.run(emphasized_signal, feed_dict={filename_placeholder: f}).flatten()

            if training:
                shift_amt = np.random.randint(-MAX_SHIFT, MAX_SHIFT)
            else:
                shift_amt = 0

            if shift_amt > 0:
                nsig = np.pad(sig, ((shift_amt, 0)), mode='constant')
            else:
                nsig = np.pad(sig, ((0, -shift_amt)), mode='constant')
                nsig = nsig[-shift_amt:]

            if sig.shape[0] > MAX_SIGNAL_LEN:
                nsig = sig[:MAX_SIGNAL_LEN]
            else:
                nsig = np.pad(sig, ((0, MAX_SIGNAL_LEN - sig.shape[0])), mode='constant')
            data = np.append(data, [nsig], axis=0)

        return data
    '''

    def _calc_mel_filterbanks(self):
        # calculate mel filterbanks
        low_mel = 1125 * math.log(1 + LOW_FREQUENCY / 700)
        high_mel = 1125 * math.log(1 + HIGH_FREQUENCY / 700)
        mel_space = np.linspace(low_mel, high_mel, NUM_MEL_FILTERBANKS + 2)
        hz_space = 700 * (np.exp(mel_space / 1125) - 1)
        filterbank_bins = np.floor((FFT_LEN + 1) * hz_space / SAMPLE_RATE)

        mel_filterbanks = np.zeros((NUM_MEL_FILTERBANKS, int(FFT_LEN / 2 + 1)))
        for m in np.arange(1, NUM_MEL_FILTERBANKS + 1):
            f_prev = filterbank_bins[m - 1]
            f = filterbank_bins[m]
            f_next = filterbank_bins[m + 1]

            for k in np.arange(int(f_prev), int(f)):
                mel_filterbanks[m - 1, k] = (k - f_prev) / (f - f_next)
            for k in np.arange(int(f), int(f_next + 1)):
                mel_filterbanks[m - 1, k] = (f_next - k) / (f_next - f)

        self.mel_fbanks = mel_filterbanks
