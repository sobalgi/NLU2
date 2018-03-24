import argparse
import os
import torch
from torch.autograd import Variable
import pickle

import data

parser = argparse.ArgumentParser(description='PyTorch Gutenberg Character level or Word level RNN/LSTM Language Model')

# Model parameters.
parser.add_argument('--model_type', type=str, default='word',
                    help='type of language model (char - character level , word - word level')
parser.add_argument('--save_dir', type=str, default='save',
                    help='folder to save/load data')
parser.add_argument('--modelname', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--corpusname', type=str,  default='corpus.pkl',
                    help='name of the file for storing corpus object')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='200',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=566,
                    help='random seed')
parser.add_argument('--startin', type=str, default='name of the king',
                    help='starting sequence of input')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=.9,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

print("Generating sentence fron %s model..."%(args.model_type))
save_dir_path = args.save_dir + '/' + args.model_type + '_' #'./' +
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(save_dir_path + args.modelname, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

if not os.path.exists(save_dir_path + args.corpusname):
    corpus = data.Corpus(args)
else:
    print("Start : Loading previously saved corpus object from save directory ...")
    corpus_file = open('./' + save_dir_path + args.corpusname, 'rb')
    corpus = pickle.load(corpus_file)
    corpus_file.close()
    print("End   : Loading previously saved corpus object from save directory ...")

ntokens = corpus.dictionary.__len__()
hidden = model.init_hidden(1)
if args.cuda:
    input.data = input.data.cuda()

if args.model_type == 'char':
    seperator = ''
    input = Variable(torch.LongTensor([corpus.dictionary.word2idx[char] for char in args.startin]), volatile=True)
else:
    seperator = ' '
    input_tkns = corpus.tknzr.tokenize(args.startin)
    input = Variable(torch.LongTensor([corpus.dictionary.word2idx[word] for word in corpus.unk_tkn_handling(input_tkns)]), volatile=True)

with open(args.outf, 'w') as outf:
    full_word = args.startin
    for i in range(args.words):
        # print(i)
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = int(torch.multinomial(word_weights, 1)[-1])
        while corpus.dictionary.idx2word[word_idx] == '_':
            word_idx = int(torch.multinomial(word_weights, 1)[-1])
        input = Variable(torch.LongTensor([word_idx]))
        word = corpus.dictionary.idx2word[word_idx]
        full_word = full_word + seperator +  word
        outf.write(word + ('\n' if i % 200 == 199 else ''))

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))

    print(full_word)
