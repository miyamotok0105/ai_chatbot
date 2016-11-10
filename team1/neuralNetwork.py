#coding:utf-8

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import math
import sys
import MeCab
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import logging
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data!='': n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        log.warning(mini_batch.shape)
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger('randomKNN')

# log.warning('Protocol problem: %s', 'connection reset')
# log.info('It works')
# log.warning('does it')
# log.debug('it really does')


train = pd.read_csv("IntentDataFormated.csv", header=0, quoting=3)

print(train.columns.values)

stop_words_ja = ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 'さ','ある', 'いる', 'も', 'する', 'から', 'な', 'こと', 'として', 'い', 'や', 'れる','など', 'なっ', 'ない', 'この', 'ため', 'その', 'あっ', 'よう', 'また', 'もの','という', 'あり', 'まで', 'られ', 'なる', 'へ', 'か', 'だ', 'これ', 'によって','により', 'おり', 'より', 'による', 'ず', 'なり', 'られる', 'において', 'ば', 'なかっ','なく', 'しかし', 'について', 'せ', 'だっ', 'その後', 'できる', 'それ', 'う', 'ので','なお', 'のみ', 'でき', 'き', 'つ', 'における', 'および', 'いう', 'さらに', 'でも','ら', 'たり', 'その他', 'に関する', 'たち', 'ます', 'ん', 'なら', 'に対して', '特に','せる', '及び', 'これら', 'とき', 'では', 'にて', 'ほか', 'ながら', 'うち', 'そして','とともに', 'ただし', 'かつて', 'それぞれ', 'または', 'お', 'ほど', 'ものの', 'に対する','ほとんど', 'と共に', 'といった', 'です', 'とも', 'ところ', 'ここ']

def text_to_clean_words( raw_text ):
    tagger = MeCab.Tagger('mecabrc')  # 別のTaggerを使ってもいい
    mecab_result = tagger.parse( raw_text )
    info_of_words = mecab_result.split('\n')
    words = []
    for info in info_of_words:
        # mecabで分けると、文の最後に’’が、その手前に'EOS'が来る
        if info == 'EOS' or info == '':
            break
            # info => 'な\t助詞,終助詞,*,*,*,*,な,ナ,ナ'
        info_elems = info.split(',')
        # 6番目に、無活用系の単語が入る。もし6番目が'*'だったら0番目を入れる
        if info_elems[6] == '*':
            # info_elems[0] => 'ヴァンロッサム\t名詞'
            words.append(info_elems[0][:-3])
            continue
        words.append(info_elems[6])
    return( " ".join( words ))         
    stops = set(stop_words_ja)
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))         

    # and return the result.
    # return( " ".join( meaningful_words ))


num_examples = train['text'].size
clean_training_data = []
for i in range( 0, num_examples ):
    if( (i+1)%10 == 0 ):
        print("Review %d of %d\n" % ( i+1, num_examples ) )
    clean_training_data.append(text_to_clean_words( train["text"][i] ) )

print("Creating the bag of words...\n")

vectorizer = CountVectorizer(analyzer = "word",   tokenizer = None,    preprocessor = None, stop_words = None,   max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_training_data)
train_data_features = train_data_features.toarray()

print(train_data_features.shape)

# Read the test data
test = pd.read_csv("testDataFormated.csv", header=0, quoting=3 )

# Verify that there are 25,000 rows and 2 columns
print(test.shape)

# Create an empty list and append the clean reviews one by one
num_test_data = len(test["text"])
clean_test_text = [] 

print("Cleaning and parsing the test set...\n")
for i in range(0,num_test_data):
    if( (i+1) % 1000 == 0 ):
        print("test %d of %d\n" % (i+1, num_test_data))
    clean_text = text_to_clean_words( test["text"][i] )
    clean_test_text.append( clean_text )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_text)
test_data_features = test_data_features.toarray()


net = Network([230, 116, 2])
net.SGD(train_data_features, 30, 10, 3.0, test_data=test_data_features)