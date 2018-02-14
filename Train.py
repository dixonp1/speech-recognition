import tensorflow as tf
import numpy as np
from Input_Data import AudioProcessor
from tensorflow.python.platform import gfile
import os.path


MAX_SHIFT = 100 #ms

'''
load file
extract features
save features in proper partition folder with name equivalent to original

'''
wav_path = "/home/patrick/PycharmProjects/Speech Recognition/speech_commands"
feature_path = "/home/patrick/PycharmProjects/Speech Recognition/test"

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
    for f in gfile.Glob(path)[:1]:
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


#def load_feature_files(feature_dir, batch_size, word_list, data_set='training'):


audio_to_feature_files(wav_path, feature_path)

'''
train and save models
'''