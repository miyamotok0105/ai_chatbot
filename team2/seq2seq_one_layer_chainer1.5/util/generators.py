import sqlite3
import MeCab

def batch(generator, batch_size):
    batch = []
    is_tuple = False
    for l in generator:
        is_tuple = isinstance(l, tuple)
        batch.append(l)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch

def sorted_parallel(generator1, generator2, pooling, order=1):
    gen1 = batch(generator1, pooling)
    gen2 = batch(generator2, pooling)
    for batch1, batch2 in zip(gen1, gen2):
        #yield from sorted(zip(batch1, batch2), key=lambda x: len(x[1]))
        for x in sorted(zip(batch1, batch2), key=lambda x: len(x[order])):
            yield x

def to_words(sentence):
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
        return list(words)

def make_input_vocab_splited():

    dbname = '../../chat_data/conversation.db'
    input_vocab_list = []
    output_vocab_list = []

    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    conversation_pairs = conn.execute("select * from conversation_pair limit 10").fetchall()
    for convs in conversation_pairs:
        input_vocab_list.append(convs[2])
        output_vocab_list.append(convs[4])
        # print("talk1: %s, talk2: %s" % (convs[2], convs[4]))
    conn.close()

    return input_vocab_list

def make_output_vocab_splited():

    dbname = '../../chat_data/conversation.db'
    input_vocab_list = []
    output_vocab_list = []

    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    conversation_pairs = conn.execute("select * from conversation_pair limit 10").fetchall()
    for convs in conversation_pairs:
        input_vocab_list.append(convs[2])
        output_vocab_list.append(convs[4])
        # print("talk1: %s, talk2: %s" % (convs[2], convs[4]))
    conn.close()

    return output_vocab_list

def input_word_list(filename):

    input_vocab_list = make_input_vocab_splited()

    for vocab_list in input_vocab_list:
        # print("vocab_list",vocab_list)
        yield to_words(vocab_list)

def output_word_list(filename):

    output_vocab_list = make_output_vocab_splited()

    for vocab_list in output_vocab_list:
        yield to_words(vocab_list)


def word_list(filename):
    with open(filename) as fp:
        for l in fp:
            yield l.split()

def letter_list(filename):
    with open(filename) as fp:
        for l in fp:
            yield list(''.join(l.split()))
