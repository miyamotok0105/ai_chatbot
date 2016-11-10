# -*- coding: utf-8 -*-
import chainer.links as L
import chainer.functions as F
from chainer import FunctionSet, Variable
from chainer import optimizers, cuda, serializers
import numpy as np

##ユニット一つのモデルを作成してみた##

model = FunctionSet(l1 = L.Linear(4, 1)) # # １つのユニット。４つの入力と１つの出力。

x_data = np.random.rand(1, 4) * 100 # 4つのランダムな配列を作成

x_data = x_data.astype(np.float32) # 変換をする必要があった

x = Variable(x_data) # Variableはキャスト

print(float(model.l1(x).data))

