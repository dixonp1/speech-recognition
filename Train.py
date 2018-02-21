import tensorflow as tf
import numpy as np
#from Input_Data import AudioProcessor
from tensorflow.python.platform import gfile
from math import floor
from glob import glob
import os.path
import random


MAX_SHIFT = 100 #ms
_SILENCE_LABEL_ = 0
_UNKN_LABEL_ = 1
_SILENCE_FILE_ = "silence.npy"

'''
load file
extract features
save features in proper partition folder with name equivalent to original

'''
wav_path = "C:\\Users\\Humphrey\\PycharmProjects\\speech-recognition\\speech_commands"
feature_path = "C:\\Users\\Humphrey\\PycharmProjects\\speech-recognition\\speech_cmd_features"
'''
def audio_to_feature_files(wav_dir, feature_dir):
    test_set_file = open(os.path.join(wav_dir, 'testing_list.txt'), mode='r')
    val_set_file = open(os.path.join(wav_dir, 'validation_list.txt'), mode='r')

    test_set = test_set_file.read().splitlines()
    val_set = val_set_file.read().splitlines()

    test_set_file.close()
    val_set_file.close()

    sess = tf.InteractiveSession()
    processor = AudioProcessor(training=True)

    path = os.path.join(wav_path, '*', '*.wav')
    for f in gfile.Glob(path):
        f_path, audio_file = os.path.split(f)
        _, word = os.path.split(f_path)

        if word == '_background_noise_':
            continue

        shift_amts = [0]
        for i in range(10):
            shift = 0
            while shift in shift_amts:
                shift = np.random.randint(-MAX_SHIFT, MAX_SHIFT)
            shift_amts.append(shift)

            if shift > 0:
                shift_padding = [shift, 0]
                offset = [0, 0]
            else:
                shift_padding = [0, -shift]
                offset = [-shift, 0]

            audio_features = sess.run(processor.features,
                                      feed_dict={processor.filename_placeholder: f,
                                                 processor.shift_amt: shift_padding,
                                                 processor.offset: offset})


            set_listing = "%s/%s" % (word, audio_file)
            folder = 0
            if set_listing in test_set:
                folder = "test"
            elif set_listing in val_set:
                folder = "validation"
            else:
                folder = "training"

            filename = audio_file.split('.wav')[0] + str(i)
            feature_file_path = os.path.join(feature_dir, folder, word, filename)
            ffdir = os.path.dirname(feature_file_path)
            if not os.path.exists(ffdir):
                os.makedirs(ffdir)

            np.save(feature_file_path, audio_features)

    sess.close()
'''

def prepare_dataset(feature_dir, word_list, silence_percent=10, unknown_percent=10):
    data = {'test': [], 'training': [], 'validation': []}

    # add file paths to words in word_list with labels
    for dataset in data:
        set_path = os.path.join(feature_dir, dataset)
        for i in range(len(word_list)):
            path = os.path.join(set_path, word_list[i], '*')
            for f in glob(path):
                data[dataset].append((f, i+2))

        # add placeholders for 'silence' files
        num_silence_files = int(len(data[dataset]) * (silence_percent/100))
        silence_path = os.path.join(feature_dir, _SILENCE_FILE_)
        for _ in range(num_silence_files):
            data[dataset].append((silence_path, _SILENCE_LABEL_))

        # add files for unrecognized words
        num_unknown_files = int((len(data[dataset]) - num_silence_files) * (unknown_percent/100))
        all_words_path = os.path.join(set_path, '*')
        all_files = []
        for p in glob(all_words_path):
            _, word = os.path.split(p)
            if word not in word_list:
                new_word = os.path.join(p, '*')
                all_files += glob(new_word)
        random.shuffle(all_files)

        for i in range(num_unknown_files):
            data[dataset].append((all_files[i], _UNKN_LABEL_))

        random.shuffle(data[dataset])

    return data

def load_batch(dataset, batch_size, batch):
    offset = batch * batch_size
    next_batch = dataset[offset:offset+batch_size]
    loaded_files = []
    for i in range(len(next_batch)):
        f = np.load(next_batch[i][0])
        loaded_files.append((f, next_batch[i][1]))

    return loaded_files

'''
d = prepare_dataset(feature_path, ["one", "two"])
test = d['test']
for item in test:
    print(item)
'''