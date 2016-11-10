#coding:utf-8
# http://gihyo.jp/dev/serial/01/machine-learning/0003 のベイジアンフィルタ実装をPython3.3向けにリーダブルに改良
import math
import sys
import MeCab
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class NaiveBayes():
    def __init__(self):
        self.vocabularies = set() #単語の集合
        self.word_count = {}  # {'花粉症対策': {'スギ花粉': 4, '薬': 2,...} }
        self.category_count = {}  # {'花粉症対策': 16, ...}

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

    def word_count_up(self, word, category):
        self.word_count.setdefault(category, {})
        self.word_count[category].setdefault(word, 0)
        self.word_count[category][word] += 1
        self.vocabularies.add(word)

    def category_count_up(self, category):
        self.category_count.setdefault(category, 0)
        self.category_count[category] += 1

    def train(self, doc, category):
        words = self.to_words(doc)
        for word in words:
            self.word_count_up(word, category)
        self.category_count_up(category)

    def prior_prob(self, category):
        num_of_categories = sum(self.category_count.values())
        num_of_docs_of_the_category = self.category_count[category]
        return num_of_docs_of_the_category / num_of_categories

    def num_of_appearance(self, word, category):
        if word in self.word_count[category]:
            return self.word_count[category][word]
        return 0

    def word_prob(self, word, category):
        # ベイズの法則の計算。通常、非常に0に近い小数になる。
        numerator = self.num_of_appearance(word, category) + 1  # +1は加算スムージングのラプラス法
        denominator = sum(self.word_count[category].values()) + len(self.vocabularies)

        # Python3では、割り算は自動的にfloatになる
        prob = numerator / denominator
        return prob

    def score(self, words, category):
        # logを取るのは、word_probが0.000....01くらいの小数になったりするため
        score = math.log(self.prior_prob(category))
        for word in words:
            score += math.log(self.word_prob(word, category))
        return score

    # logを取らないと値が小さすぎてunderflowするかも。
    def score_without_log(self, words, category):
        score = self.prior_prob(category)
        for word in words:
            score *= self.word_prob(word, category)
        return score

    def classify(self, doc):
        best_guessed_category = None
        max_prob_before = -sys.maxsize
        words = self.to_words(doc)

        for category in self.category_count.keys():
            prob = self.score(words, category)
            if prob > max_prob_before:
                max_prob_before = prob
                best_guessed_category = category
        return best_guessed_category

if __name__ == '__main__':

    nb = NaiveBayes()

    # for line in open("../text/it-life-hack/it-life-hack-6590267.txt", "r"):
    #     nb.train(line,'IT')

    load_File_main_count = 20  #2〜700を使用すること
    load_File_count = 1
    list = os.listdir("../text/movie-enter/")
    for file in list:
        for line in open("../text/movie-enter/" + file, "r"):
            nb.train(line,'MOVIE')
            load_File_count = load_File_count + 1

            if load_File_count == load_File_main_count:
                load_File_count = 1
                break

    list = os.listdir("../text/it-life-hack/")
    for file in list:
        for line in open("../text/it-life-hack/" + file, "r"):
            nb.train(line,'IT')
            load_File_count = load_File_count + 1

            if load_File_count == load_File_main_count:
                load_File_count = 1
                break

    list = os.listdir("../text/topic-news/")
    for file in list:
        for line in open("../text/topic-news/" + file, "r"):
            nb.train(line,'NEWS')
            load_File_count = load_File_count + 1

            if load_File_count == load_File_main_count:
                load_File_count = 1
                break

    list = os.listdir("../text/sports-watch/")
    for file in list:
        for line in open("../text/sports-watch/" + file, "r"):
            nb.train(line,'SPORTS')
            load_File_count = load_File_count + 1

            if load_File_count == load_File_main_count:
                load_File_count = 1
                break

    list = os.listdir("../text/smax/")
    for file in list:
        for line in open("../text/smax/" + file, "r"):
            nb.train(line,'PC')
            load_File_count = load_File_count + 1

            if load_File_count == load_File_main_count:
                load_File_count = 1
                break

    list = os.listdir("../text/kaden-channel/")
    for file in list:
        for line in open("../text/kaden-channel/" + file, "r"):
            nb.train(line,'KADEN')
            load_File_count = load_File_count + 1

            if load_File_count == load_File_main_count:
                load_File_count = 1
                break


#    doc = '写真を切り貼りしてスクラップブックを作りたい人'
    doc = '今日はきゃりーぱみゅぱみゅ3rdアルバム「ピカピカふぁんたじん」のフラゲ日です！ 約1年ぶりのアルバムになりますが、今回はキラキラとかピカピカがコンセプトになっています。ふぁんたじんとゆうのは「ふぁんたじーな人」とゆう意味で中田ヤスタカさんがニヤニヤしながら「きゃりーはふぁんたじん」だからねって名付けてくれました。気に入ってるタイトルです'
    parse_doc = nb.to_words(doc)
    print('%s => 推定カテゴリ: %s' % (doc, nb.classify(doc)))
#    print(parse_doc)

    #計算がおかしいので修正する必要あり
    print('MOVIEカテゴリである確率: %s' % nb.score_without_log(parse_doc, 'MOVIE'))
    print('ITカテゴリである確率: %s' % nb.score_without_log(parse_doc, 'IT'))
    print('NEWSカテゴリである確率: %s' % nb.score_without_log(parse_doc, 'NEWS'))
    print('SPORTSカテゴリである確率: %s' % nb.score_without_log(parse_doc, 'SPORTS'))
    print('PCカテゴリである確率: %s' % nb.score_without_log(parse_doc, 'PC'))
    print('KADENカテゴリである確率: %s' % nb.score_without_log(parse_doc, 'KADEN'))

    #以下グラフにして出そうと頑張り中
    # for k,v in nb.word_count.items():
    #     print(k)
#        plt.plot(k,v)


#    plt.show()

    # print(nb.vocabularies)
    # print(nb.word_count)
    # print(nb.category_count)

    # plt.plot(nb.category_count)


