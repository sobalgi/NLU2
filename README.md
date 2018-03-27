# NLU2
LSTM Based Character and Word Level Language Models

Corpus : NLTK Gutenberg

1. Generate Character and Word level language model using PyTorch (CUDA enabled for gpu optimization)

example :
a. python3.6 main.py --cuda --model_type 'char' --emsize 128 --nhid 128 --bptt 50
b. python3.6 main.py --cuda --model_type 'word' --emsize 300 --nhid 300 --bptt 10

More help : python main.py -h

PyTorch Gutenberg Character level or Word level RNN/LSTM Language Model

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        type of language model (char - character level , word
                        - word level
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --train_percent TRAIN_PERCENT
                        training size percent
  --dev_percent DEV_PERCENT
                        development/validation size percent
  --save_dir SAVE_DIR   folder to save/load data
  --modelname MODELNAME
                        path to save the final model
  --corpusname CORPUSNAME
                        name of the file for storing corpus object
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --bptt BPTT           sequence length
  --batch_size N        batch size
  --lr LR               initial learning rate
  --lr_decay LR_DECAY   decay factor for learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval


2. Generate the sentences from saved models

example: 
a. python3.6 generate_sentence.py --cuda --model_type 'char' --words 100
b. python3.6 generate_sentence.py --cuda --model_type 'word' --words 20

PyTorch Gutenberg Character level or Word level RNN/LSTM Language Model

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        type of language model (char - character level , word
                        - word level
  --save_dir SAVE_DIR   folder to save/load data
  --modelname MODELNAME
                        path to save the final model
  --corpusname CORPUSNAME
                        name of the file for storing corpus object
  --outf OUTF           output file for generated text
  --words WORDS         number of words to generate
  --seed SEED           random seed
  --startin STARTIN     starting sequence of input
  --cuda                use CUDA
  --temperature TEMPERATURE
                        temperature - higher will increase diversity
  --log-interval LOG_INTERVAL
                        reporting interval
