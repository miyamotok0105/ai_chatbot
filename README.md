# ai_chatbot

todo:全体的に整理する。

## 概要

1.雑談と質問を分類するフィルター
2.雑談を返す


## 環境

mac

    brew install mecab
    brew install mecab-ipadic
    sudo pip install mecab-python3
    sudo pip install chainer==1.12.0
    sudo pip install h5py


## 雑談返答


    cd team2/seq2seq_one_layer_chainer1.5
    python mt_s2s_encdec.py train ./models/out

