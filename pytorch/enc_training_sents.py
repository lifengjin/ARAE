from models import load_models, generate
from utils import Dictionary, Corpus, batchify
import argparse
import numpy as np
import torch
import dill as pickle
from torch.autograd import Variable
from collections import namedtuple
parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')
parser.add_argument('--load_path', type=str, required=True,
                    help='directory to load models from')
parser.add_argument('--inf', type=str, required=True,
                    help='filename and path where the corpus is')
args = parser.parse_args()
print(vars(args))

class EncodeCopus(Corpus):
    def __init__(self, path, maxlen, vocab_size=11000, lowercase=False, dictionary=None):
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        if maxlen  == -1:
            self.maxlen = np.inf
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.train_path = path
        self.text = []
        self.hiddens = []
        self.labels = []
        # make the vocabulary from training set
        if dictionary is None:
            self.make_vocab()
        else:
            self.dictionary = dictionary

        self.train = self.tokenize(self.train_path)
        self.Item = namedtuple('Item', ['text', 'hidden', 'label'])

    def get_batches(self):
        batchsize = 1
        data = self.train
        return batchify(data, batchsize, shuffle=False, gpu=True)

    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                line = line.strip().split('\t')
                if self.lowercase:
                    words = line[0].lower().strip().split(" ")
                else:
                    words = line[0].strip().split(" ")
                if len(words) > self.maxlen:
                    dropped += 1
                    continue
                words = ['<sos>'] + words
                words += ['<eos>']
                # vectorize
                vocab = self.dictionary.word2idx
                unk_idx = vocab['<oov>']
                self.text.append(words)
                try:
                    self.labels.append(line[1])
                except:
                    print(line)
                    raise
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                lines.append(indices)

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines

    def __index__(self, i):
        return self.Item(self.text[i], self.hiddens[i], self.labels[i])

model_args, idx2word, autoencoder, gan_gen, gan_disc \
        = load_models(args.load_path)
# print(idx2word)

word2idx = {word : index for index, word in idx2word.items()}

dic = Dictionary()
dic.word2idx = word2idx
dic.idx2word = idx2word

corpus = EncodeCopus(args.inf, maxlen=-1, dictionary=dic)

batches = corpus.get_batches()
autoencoder.cuda()
autoencoder.eval()

with open(args.inf + '.corpus', 'wb') as b:
    hiddens = []
    for index, (source, target, length) in enumerate(batches):
        # print(source, length)
        hidden = autoencoder.encode(Variable(source), length,None)
        # print(hidden)
        hiddens.append( hidden.data.cpu())
    corpus.hiddens = hiddens
    pickle.dump(corpus, b)