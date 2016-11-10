# coding: utf-8
import numpy as np
from numpy import hstack, vsplit, hsplit, reshape
from sklearn.cross_validation import train_test_split
from collections import defaultdict
import six
import sys
import chainer
import chainer.links as L
from chainer import optimizers, cuda, serializers
import chainer.functions as F
import argparse
from gensim import corpora, matutils
import MeCab
from matplotlib import pyplot as plt


"""
単純なNNでテキスト分類 (posi-nega)
 - 隠れ層は2つ
 - 文書ベクトルにはBoWモデルを使用
 @author ichiroex
"""

class InitData():

    def __init__(self):
        source = []
        target = []
        comment = []
        word = []

    def to_words(self, sentence):
        """
        入力: 'すべて自分のほうへ'
        出力: tuple(['すべて', '自分', 'の', 'ほう', 'へ'])
        """
        tagger = MeCab.Tagger('mecabrc')  # 別のTaggerを使ってもいい
        mecab_result = tagger.parse(sentence)
        info_of_words = mecab_result.split('\n')
        words = []
        for info in info_of_words:
            # macabで分けると、文の最後に’’が、その手前に'EOS'が来る
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
        return tuple(words)

    def load_data(self, fname):

        source = []
        target = []
        comment = []
        word = []
        f = open(fname, "r")

        document_list = [] #各行に一文書. 文書内の要素は単語
        for l in f.readlines():
            sample = l.strip().split(" ", 1)        #ラベルと単語列を分ける
            # print(sample)
            label = int(sample[0])                  #ラベル
            target.append(label)
            # document_list.append(sample[1].split()) #単語分割して文書リストに追加
            document_list.append(self.to_words(sample[1])) #単語分割して文書リストに追加
            comment.append(sample[1])
            word.append(self.to_words(sample[1]))
            # print(sample[1])


        print(document_list)

        #単語辞書を作成
        dictionary = {}   
        dictionary = corpora.Dictionary(document_list)
        dictionary.filter_extremes(no_below=0, no_above=100) 
        # no_below: 使われている文書がno_below個以下の単語を無視
        # no_above: 使われてる文章の割合がno_above以上の場合無視
        
        #文書のベクトル化
        for document in document_list:
            tmp = dictionary.doc2bow(document) #文書をBoW表現
            vec = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0]) 
            source.append(vec)


        dataset = {}
        dataset['source'] = np.array(source)
        dataset['target'] = np.array(target)
        dataset['comment'] = comment;
        dataset['word'] = word;
        print("------------")
        # print(dataset['source'])
        # print(dataset['target'])
        print(dataset['comment'])
        print("全データ数:", len(dataset['source'])) # データの数
        print ("辞書に登録された単語数:", len(dictionary.items())) # 辞書に登録された単語の数

        return dataset, dictionary

# 工事中。いずれこっちの記法に変換予定
# ※１
# class FFNN_Net(chainer.Chain):

#     def __init__(self):
#         """
#             モデルの定義
#         """
#         super(FFNN_Net, self).__init__(
#             l1=L.Linear(in_units, n_units),
#             l2=L.Linear(n_units, n_units),
#             l3=L.Linear(n_units,  n_label)
#         )
#         self.train = True

#     def __call__(self, x, t, train=True):
#         """
#             forword処理
#         """
#         h1 = F.relu(model.l1(x))
#         h2 = F.relu(model.l2(h1))
#         y = model.l3(h2)

#         return F.softmax_cross_entropy(y, t), F.accuracy(y, t), y.data



if __name__ == '__main__':

    #学習グラフ用
    losses =[]
    accuracies =[]

    initdata = InitData()

    #引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu  '    , dest='gpu'        , type=int, default=0,            help='1: use gpu, 0: use cpu')
    parser.add_argument('--data '    , dest='data'       , type=str, default='input.dat',  help='an input data file')
    parser.add_argument('--epoch'    , dest='epoch'      , type=int, default=100,          help='number of epochs to learn')
    parser.add_argument('--batchsize', dest='batchsize'  , type=int, default=1,           help='learning minibatch size')
    parser.add_argument('--units'    , dest='units'      , type=int, default=100,           help='number of hidden unit')

    args = parser.parse_args()

    batchsize   = args.batchsize    # minibatch size
    n_epoch     = args.epoch        # エポック数(パラメータ更新回数)

    # Prepare dataset
    dataset, dictionary = initdata.load_data(args.data)

    dataset['source'] = dataset['source'].astype(np.float32) #文書ベクトル
    dataset['target'] = dataset['target'].astype(np.int32)   #ラベル

    x_train, x_test, y_train, y_test, c_train, c_test, word_train, word_test = train_test_split(dataset['source'], dataset['target'], dataset['comment'], dataset['word'], test_size=0.001)

    print("------------")
    # print("x_train", x_train)
    # print("x_test", x_test)
    # print(y_train)
    # print(y_test)

    N_test = y_test.size         # test data size
    N = len(x_train)             # train data size
    in_units = x_train.shape[1]  # 入力層のユニット数 (語彙数)
    print("学習データ数:", N)

    n_units = args.units # 隠れ層のユニット数
    n_label = 2          # 出力層のユニット数

    #モデルの定義
    model = chainer.Chain(l1=L.Linear(in_units, n_units),
                          l2=L.Linear(n_units, n_units),
                           l3=L.Linear(n_units,  n_label))
    # ※１
    # model = FFNN_Net()

    # #GPUを使うかどうか
    # if args.gpu > 0:
        # cuda.check_cuda_available()
        # cuda.get_device(args.gpu).use()
        # model.to_gpu()
        # xp = np if args.gpu <= 0 else cuda.cupy #args.gpu <= 0: use cpu, otherwise: use gpu

    xp = np

    batchsize = args.batchsize
    n_epoch   = args.epoch

    def forward(x, t, train=True):
        h1 = F.relu(model.l1(x))
        h2 = F.relu(model.l2(h1))
        y = model.l3(h2)
        # h1 = F.dropout(F.relu(model.l1(x)), train=True)
        # h2 = F.dropout(F.relu(model.l2(h1)), train=True)
        # y = model.l3(h2)
        # h1 = F.dropout(F.relu(model.l1(x)))
        # h2 = F.dropout(F.relu(model.l2(h1)))
        # y = model.l3(h2)


        # print("l1:",h1.data.size)
        # print("l2:",h2.data.size)
        # print("l3:",y.data.size)
        # print(y.data)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t), y.data

    # Setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # Learning loop---------------------------------------------
    for epoch in six.moves.range(1, n_epoch + 1):

        print ('epoch', epoch)

        # training---------------------------------------------
        perm = np.random.permutation(N) #ランダムな整数列リストを取得
        sum_train_loss     = 0.0
        sum_train_accuracy = 0.0
        for i in six.moves.range(0, N, batchsize):

            #perm を使い x_train, y_trainからデータセットを選択 (毎回対象となるデータは異なる)
            x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]])) #source
            t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]])) #target
            c = chainer.Variable(xp.asarray(c_train[perm[i:i + batchsize]]))
            w = chainer.Variable(xp.asarray(word_train[perm[i:i + batchsize]]))

            # print("x_train ",x_train)
            # print("x_train ",np.vsplit(x_train, 2))

            model.zerograds()            # 勾配をゼロ初期化
            loss, acc, y = forward(x, t)    # 順伝搬
            # sum_train_loss      += float(cuda.to_cpu(loss.data)) * len(t)   # 平均誤差計算用
            # sum_train_accuracy  += float(cuda.to_cpu(acc.data )) * len(t)   # 平均正解率計算用
            sum_train_loss = float(cuda.to_cpu(loss.data))
            sum_train_accuracy = float(cuda.to_cpu(acc.data ))

            loss.backward()              # 誤差逆伝播
            optimizer.update()           # 最適化：重み、バイアスの更新 
            losses.append(loss.data) #誤差関数グラフ用


        # print('train mean loss={}, accuracy={}'.format(sum_train_loss / N, sum_train_accuracy / N)) #平均誤差
        print('train mean loss={}, accuracy={}'.format(sum_train_loss, sum_train_accuracy)) #誤差

        print(np.argmax(y), y, c.data, w.data) #出力結果一覧
        

        # evaluation---------------------------------------------
        sum_test_loss     = 0.0
        sum_test_accuracy = 0.0
        for i in six.moves.range(0, N_test, batchsize):

            # all test data
            x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]))
            t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]))
            c = chainer.Variable(xp.asarray(c_train[i:i + batchsize]))
            w = chainer.Variable(xp.asarray(word_train[i:i + batchsize]))

            loss, acc, y = forward(x, t, train=False)

            # sum_test_loss     += float(cuda.to_cpu(loss.data)) * len(t)
            # sum_test_accuracy += float(cuda.to_cpu(acc.data))  * len(t)
            sum_test_loss     = float(cuda.to_cpu(loss.data))
            sum_test_accuracy = float(cuda.to_cpu(acc.data))
            accuracies.append(acc.data) #汎化性能グラフ用

        # print(' test mean loss={}, accuracy={}'.format(
        #     sum_test_loss / N_test, sum_test_accuracy / N_test)) #平均誤差
        print(' test mean loss={}, accuracy={}'.format(sum_test_loss, sum_test_accuracy)) #誤差
        print(np.argmax(y), y, c.data, w.data) #出力結果一覧
        print (type(y), type(np.argmax(y)))
        print("-----------------------------------")


    #modelとoptimizerを保存---------------------------------------------
    print ('save the model')
    serializers.save_npz('pn_classifier_ffnn2.model', model)
    print ('save the optimizer')
    serializers.save_npz('pn_classifier_ffnn2.state', optimizer)

    plt.plot(losses, label = "sum_train_loss")
    plt.plot(accuracies, label = "sum_train_accuracy")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


    # model = serializers.load_npz('pn_classifier_ffnn.model', optimizer)
    # serializers.load_npz('pn_classifier_ffnn.state', optimizer)




