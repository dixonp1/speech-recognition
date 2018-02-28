import tensorflow as tf
import numpy as np
#from Input_Data import AudioProcessor
from NeuralNetwork import model
import post_processing as phm
from tensorflow.python.platform import gfile
from math import floor
from glob import glob
import os.path
import random
import time as t


MAX_SHIFT = 100 #ms
_SILENCE_LABEL_ = 0
_UNKN_LABEL_ = 1
_SILENCE_FILE_ = "silence.npy"

# windows
wav_path = "C:\\Users\\Humphrey\\PycharmProjects\\speech-recognition\\speech_commands"
feature_path = "C:\\Users\\Humphrey\\PycharmProjects\\speech-recognition\\speech_cmd_features"
save_path = "C:\\Users\\Humphrey\\PycharmProjects\\speech-recognition\\models"

# linux
#wav_path = "/home/patrick/PycharmProjects/Speech Recognition/speech_commands"
#feature_path = "/home/patrick/PycharmProjects/Speech Recognition/speech_cmd_features"

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
                                                 processor.volume_placeholder: 1,
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
            

    # save placeholder file for 'silence'
    path = os.path.join(wav_path, 'bed', '0a7c2a8d_nohash_0.wav')
    audio_features = sess.run(processor.features,
                              feed_dict={processor.filename_placeholder: path,
                                         processor.volume_placeholder: 0,
                                         processor.shift_amt: [0, 0],
                                         processor.offset: [0, 0]})
    silence_path = os.path.join(feature_dir, _SILENCE_FILE_)
    np.save(silence_path, audio_features)
    
    sess.close()
'''

def prepare_dataset(feature_dir, word_list, silence_percent=10, unknown_percent=10):
    data = {'test': [], 'training': [], 'validation': []}

    num_examples = len(word_list) * 15000
    # add file paths to words in word_list with labels
    for dataset in data:
        set_path = os.path.join(feature_dir, dataset)
        for i in range(len(word_list)):
            label = np.zeros(len(word_list) + 2)
            label[i+2] = 1
            path = os.path.join(set_path, word_list[i], '*')
            for f in glob(path)[:num_examples]:
                data[dataset].append((f, label))

        set_size = len(data[dataset])
        # add placeholders for 'silence' files
        num_silence_files = int(set_size * (silence_percent/100))
        silence_path = os.path.join(feature_dir, _SILENCE_FILE_)
        label = np.zeros(len(word_list) + 2)
        label[_SILENCE_LABEL_] = 1
        for _ in range(num_silence_files):
            data[dataset].append((silence_path, label))

        # add files for unrecognized words
        num_unknown_files = int(set_size * (unknown_percent/100))
        all_words_path = os.path.join(set_path, '*')
        all_files = []
        for p in glob(all_words_path):
            _, word = os.path.split(p)
            if word not in word_list:
                new_word = os.path.join(p, '*')
                all_files += glob(new_word)
        random.shuffle(all_files)

        label = np.zeros(len(word_list) + 2)
        label[_UNKN_LABEL_] = 1
        for i in range(num_unknown_files):
            data[dataset].append((all_files[i], label))

        random.shuffle(data[dataset])

    return data

def load_batch(dataset, batch, batch_size):
    offset = batch * batch_size
    if batch_size == -1:
        next_batch = dataset
    else:
        next_batch = dataset[offset:offset+batch_size]
    loaded_files = []
    labels = []
    for i in range(len(next_batch)):
        f = np.load(next_batch[i][0])
        loaded_files.append(f)
        labels.append(next_batch[i][1])

    return [loaded_files, labels]

start = t.time()

batch_size = 64
epoch = 100
word_list = ["one", "two", "three"]
data = prepare_dataset(feature_path, word_list)

sig_features = tf.placeholder(tf.float32, [None, 99, 40])
network = model(sig_features, len(word_list) + 2)

# cost function
#predictions = tf.placeholder(tf.float32, [None, len(word_list) + 2])
predictions = network.softmax
labels = tf.placeholder(tf.float32, [None, len(word_list) + 2])
#cross_entropy = labels * tf.log(predictions) + (1 - labels) * tf.log(1 - predictions)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predictions)
cross_entropy = tf.reduce_mean(cross_entropy)

# training optimizer
with tf.control_dependencies([tf.add_check_numerics_ops()]):
    #train = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
    train = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# accuracy
correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

saver = tf.train.Saver(tf.global_variables())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# run graph
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())

for i in range(150):
    print('epoch', i)
    random.shuffle(data['training'])
    num_training_batches = int(len(data['training']) / batch_size)
    #num_validation_batches = int(len(data['validatoin']) / batch_size)
    if i % 10 == 0:
        random.shuffle(data['validation'])
        validation = load_batch(data['validation'], 0, -1)
        #val_pred = model.forward_prop(val_batch[0], sess)
        train_accuracy = accuracy.eval(feed_dict={
            sig_features: validation[0], labels: validation[1]
        })
        print('step %d\ttraining accuracy: %g' % (i, train_accuracy))

    for b in range(num_training_batches):
        batch = load_batch(data['training'], b, batch_size)
        #batch_pred = model.forward_prop(batch[0], sess)
        train.run(feed_dict={sig_features: batch[0], labels: batch[1]})

test_data = load_batch(data['test'], 0, -1)
#test_pred = model.forward_prop(test_data[0], sess)
test_accuracy = accuracy.eval(feed_dict={sig_features: test_data[0], labels: test_data[1]})
print('test accuracy %g' % test_accuracy)

filename = "norm_pool_%g.ckpt" % test_accuracy
cp_path = os.path.join(save_path, filename)
saver.save(sess, cp_path)

end = t.time()
print('runtime: ', end-start)
sess.close()
'''
d = prepare_dataset(feature_path, ["one", "two"])
test = d['test']
for item in test:
    print(item)
'''