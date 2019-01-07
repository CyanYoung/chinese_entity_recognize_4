import pickle as pk

import numpy as np

from keras.models import load_model

from keras.preprocessing.sequence import pad_sequences

from represent import add_buf

from nn_arch import cnn, rnn

from util import map_item


def ind2label(label_inds):
    ind_labels = dict()
    for word, ind in label_inds.items():
        ind_labels[ind] = word
    return ind_labels


seq_len = 50

path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

funcs = {'cnn': cnn,
         'rnn': rnn}

paths = {'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5'}

models = {'cnn': load_model(map_item('cnn', paths)),
          'rnn': load_model(map_item('rnn', paths))}


def predict(text, name):
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    if name == 'cnn':
        pad_seq = add_buf(pad_seq)
    model = map_item(name, models)
    probs = model.predict(pad_seq)[0]
    inds = np.argmax(probs, axis=1)
    preds = [ind_labels[ind] for ind in inds[-len(text):]]
    pairs = list()
    for word, pred in zip(text, preds):
        pairs.append((word, pred))
    return pairs


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('cnn: %s' % predict(text, 'cnn'))
        print('rnn: %s' % predict(text, 'rnn'))
