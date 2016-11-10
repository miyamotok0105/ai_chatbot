#coding:utf-8
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
from sklearn.ensemble import RandomForestClassifier
import autosklearn.classification


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
log = logging.getLogger('autosklearn')

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
        print("Review %d of %d\n" % ( i+1, num_examples ))
    clean_training_data.append(text_to_clean_words( train["text"][i] ) )

print("Creating the bag of words...\n")

vectorizer = CountVectorizer(analyzer = "word",   tokenizer = None,    preprocessor = None, stop_words = None,   max_features = 5000) 

train_data_features = vectorizer.fit_transform(clean_training_data)
train_data_features = train_data_features.toarray()

print("train_data_features.shape:",train_data_features.shape)
print("clean_training_data:",clean_training_data)
print("train_data_features:",train_data_features)
print("train[label]:",train["label"])


print("Training the autosklearn...")

# Initialize a Random Forest classifier with 100 trees
cls = RandomForestClassifier(n_estimators = 10) 
# cls = autosklearn.classification.AutoSklearnClassifier()
⬆︎ここをautoにしたいけど、エラーになる。


# Fit the autosklearn to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
cls = cls.fit( train_data_features, train["label"] )

# print('accuracy on the training set: %f' %cls.score( train_data_features, train["label"] ))

# # Read the test data
# test = pd.read_csv("testDataFormated.csv", header=0, quoting=3 )

# # Verify that there are 25,000 rows and 2 columns
# print(test.shape)

# # Create an empty list and append the clean reviews one by one
# num_test_data = len(test["text"])
# clean_test_text = [] 

# print("Cleaning and parsing the test set...\n")
# for i in range(0,num_test_data):
#     if( (i+1) % 1000 == 0 ):
#         print("test %d of %d\n" % (i+1, num_test_data))
#     clean_text = text_to_clean_words( test["text"][i] )
#     clean_test_text.append( clean_text )

# # Get a bag of words for the test set, and convert to a numpy array
# test_data_features = vectorizer.transform(clean_test_text)
# test_data_features = test_data_features.toarray()

# # Use the random cls to make sentiment label predictions
# result = cls.predict(test_data_features)

# # # Copy the results to a pandas dataframe with an "id" column and
# # # a "sentiment" column
# # output = pd.DataFrame( data={"text":clean_test_text, "label":result, 'correct':test['label']} )

# # # Use pandas to write the comma-separated output file
# # output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

# goodPrediction = 0
# badPrediction = 0

# for i in range(0,num_test_data):
#     if result[i] == test['label'][i]:
#         goodPrediction += 1
#     else:
#         badPrediction += 1
#         print('sample = %s pred = %s, correct = %s confidence = %f' % (test['text'][i], result[i] , test['label'][i], np.amax(forest.predict_proba(train_data_features[i].reshape(1,-1)))))
# print('Accuracy = %f' % (1.0*goodPrediction/ (badPrediction + goodPrediction)))


