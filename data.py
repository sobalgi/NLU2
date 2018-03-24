import torch
import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import TweetTokenizer
from collections import Counter
import re
import time
import json
import os
import pickle

# import argparse
# parser = argparse.ArgumentParser(description='PyTorch Gutenberg Character level or Word level RNN/LSTM Language Model')
# parser.add_argument('--model_type', type=str, default='char',
#                     help='type of language model (char - character level , word - word level')
# parser.add_argument('--model', type=str, default='LSTM',
#                     help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
# parser.add_argument('--train_percent', type=float, default=.8,
#                     help='training size percent')
# parser.add_argument('--dev_percent', type=float, default=.1,
#                     help='development/validation size percent')
# parser.add_argument('--save_dir', type=str, default='save',
#                     help='folder to save/load data')
# parser.add_argument('--modelname', type=str,  default='model.pt',
#                     help='path to save the final model')
# parser.add_argument('--corpusname', type=str,  default='corpus.pkl',
#                     help='name of the file for storing corpus object')
# parser.add_argument('--emsize', type=int, default=80,
#                     help='size of word embeddings')
# parser.add_argument('--nhid', type=int, default=80,
#                     help='number of hidden units per layer')
# parser.add_argument('--nlayers', type=int, default=2,
#                     help='number of layers')
# parser.add_argument('--bptt', type=int, default=50,
#                     help='sequence length')
# parser.add_argument('--batch_size', type=int, default=50, metavar='N',
#                     help='batch size')
# parser.add_argument('--lr', type=float, default=4,
#                     help='initial learning rate')
# parser.add_argument('--lr_decay', type=float, default=.8,
#                     help='decay factor for learning rate')
# parser.add_argument('--clip', type=float, default=0.25,
#                     help='gradient clipping')
# parser.add_argument('--epochs', type=int, default=20,
#                     help='upper epoch limit')
# parser.add_argument('--dropout', type=float, default=0.2,
#                     help='dropout applied to layers (0 = no dropout)')
# parser.add_argument('--tied', action='store_true',
#                     help='tie the word embedding and softmax weights')
# parser.add_argument('--seed', type=int, default=1111,
#                     help='random seed')
# parser.add_argument('--cuda', action='store_true',
#                     help='use CUDA')
# parser.add_argument('--log-interval', type=int, default=200, metavar='N',
#                     help='report interval')
# args = parser.parse_args()

# Dictionary class
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, args):
        super(Corpus, self).__init__()

        # Threshold count for unknown assignment
        self.threshold_unk_count = 2
        # Unknown token
        self.unk_tkn = '_'

        save_dir = args.save_dir
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        start_time = time.time()
        if not os.path.exists(self.save_dir + '/' + 'gutenberg.json'):
            print('Pre-processing Gutenberg Corpus from nltk ...')

            self.train_percent = args.train_percent
            self.dev_percent = args.dev_percent
            train_txt = ''
            dev_txt = ''
            test_txt = ''
            for i,fid in enumerate(gutenberg.fileids()):
                train,dev,test = self.tokenize(fid)
                train_txt = ' '.join([train_txt, train])
                dev_txt = ' '.join([dev_txt, dev])
                test_txt = ' '.join([test_txt, test])

            json_content = json.dumps({
                'train_data_txt': train_txt,
                'dev_data_txt': dev_txt,
                'test_data_txt': test_txt
            }, indent=4)

            with open(self.get_file_name('gutenberg.json'), 'w') as f:
                print('Saving Pre-processed Gutenberg Corpus to gutenberg.json ...')
                f.write(json_content)

        else:
            print('Loading Pre-processed Gutenberg Corpus from gutenberg.json ...')
            with open(self.get_file_name('gutenberg.json'), 'r') as f:
                data_dict = json.load(f)
                train_txt = data_dict['train_data_txt']
                dev_txt = data_dict['dev_data_txt']
                test_txt = data_dict['test_data_txt']
                del data_dict

        self.dictionary = Dictionary()
        if args.model_type == 'char':
            for word in sorted(set(train_txt)):
                _ = self.dictionary.add_word(word)

            self.train_len = len(train_txt)
            self.dev_len = len(dev_txt)
            self.test_len = len(test_txt)
            self.train = torch.LongTensor([self.dictionary.word2idx[c] for c in train_txt])
            self.dev = torch.LongTensor([self.dictionary.word2idx[c] for c in dev_txt])
            self.test = torch.LongTensor([self.dictionary.word2idx[c] for c in test_txt])

        else:
            self.tknzr = TweetTokenizer()
            train_tkns = self.tknzr.tokenize(train_txt)
            dev_tkns = self.tknzr.tokenize(dev_txt)
            test_tkns = self.tknzr.tokenize(test_txt)


            self.train_len = len(train_tkns)
            self.dev_len = len(dev_tkns)
            self.test_len = len(test_tkns)

            # Unknown Token Handling
            self.local_dict = Counter(train_tkns)
            train_tkns = self.unk_tkn_handling(train_tkns)
            dev_tkns= self.unk_tkn_handling(dev_tkns)
            test_tkns = self.unk_tkn_handling(test_tkns)

            # train_tkns_new = self.unk_tkn_handling(train_tkns, local_dict)
            # dev_tkns_new = self.unk_tkn_handling(dev_tkns, local_dict)
            # test_tkns_new = self.unk_tkn_handling(test_tkns, local_dict)

            for word in train_tkns:
                _ = self.dictionary.add_word(word)

            self.train = torch.LongTensor([self.dictionary.word2idx[c] for c in train_tkns])
            self.dev = torch.LongTensor([self.dictionary.word2idx[c] for c in dev_tkns])
            self.test = torch.LongTensor([self.dictionary.word2idx[c] for c in test_tkns])

        self.save_object(self, self.get_file_name(args.model_type + '_' + args.corpusname))

        print('Dictionary              : %s' % self.dictionary.idx2word)
        print('Total Dictionary Length : %d' % self.dictionary.__len__())
        print('Total Train data Length : %d' % self.train_len)
        print('Total Dev data Length   : %d' % self.dev_len)
        print('Total Test data Length  : %d' % self.test_len)
        end_time = time.time()
        print('Finished %s model in %d minutes %d seconds' % (args.model_type,(end_time - start_time) / 60, (end_time - start_time) % 60))

    def tokenize(self, fid):
        data = gutenberg.raw(fileids = fid)
        # data = data.lower()
        # data = re.sub('(\\n)+', ' ', data)
        data = re.sub('--', '-', data)
        data = re.sub('\.+', '.', data)
        data = re.sub('[^ 0-9a-zA-Z?!:;\,\.\-\'\"\\n]', ' _ ', data) # _ is unknown letter or word
        data = re.sub('(\s+&[^\\n])', ' ', data)
        # data = re.sub("([@]+)|([-]{2,})|(\\n)|([\s]{2,})|([\"*:\[\]\(\)]+)", ' ', data)
        sents = nltk.tokenize.sent_tokenize(data)
        total_sents = len(sents)
        train_size = int(total_sents*self.train_percent)
        dev_size = int(total_sents*self.dev_percent)

        return ' '.join(sents[:train_size-1]), ' '.join(sents[train_size:train_size+dev_size-1]), ' '.join(sents[train_size+dev_size:])

    def get_file_name(self, file_name):
        return self.save_dir + '/' + file_name

    @staticmethod
    def save_object(obj, file_name):
        with open('./'+file_name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def unk_tkn_handling(self, data_tkns):
        return [self.unk_tkn if self.local_dict[word] < self.threshold_unk_count else word for word in data_tkns]

# dummy = Corpus(args)