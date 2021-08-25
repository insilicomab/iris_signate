# -*- coding: utf-8 -*-


'''
データの読み込みと確認
'''

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ランダムシードの設定
import random
np.random.seed(1234)
random.seed(1234)

# データの読み込み
train = pd.read_csv('./data/train.tsv', sep='\t')
test = pd.read_csv('./data/test.tsv', sep='\t')
submission = pd.read_csv('./data/sample_submit.csv')

# データの確認
print(train.head())
print(train.dtypes)

'''
特徴量エンジニアリング
'''

# 説明変数と目的変数を指定
X_train = train.drop(['class', 'id'], axis=1)
Y_train = train['class']

# ラベルデータをone-hotベクトルに直す
labels = {
    'Iris-setosa': [1, 0, 0], 
    'Iris-versicolor': [0, 1, 0], 
    'Iris-virginica': [0, 0, 1]
}


Y_train_nums = np.array(list(map(lambda v : labels[v] , Y_train)))
X_train_nums = np.array(X_train)

'''
モデルの構築と評価
'''

# ライブラリのインポート
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

# 学習用とテスト用に分割する
x_train, x_valid, y_train, y_valid = train_test_split(X_train_nums, Y_train_nums,
                                                    train_size=0.8,
                                                    random_state=0,
                                                    stratify=Y_train_nums)

# モデル構造を定義
Dense = keras.layers.Dense
model = keras.models.Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(3, activation='softmax'))

# モデルを構築
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

# 学習を実行
model.fit(x_train, y_train,
    batch_size=20,
    epochs=1000)

# モデルを評価
score = model.evaluate(x_valid, y_valid, verbose=1)
print('正解率=', score[1], 'loss=', score[0])

'''
テストデータの予測
'''

# 説明変数を指定
X_test = test.drop('id', axis=1)
X_test_nums = np.array(X_test)

# テストデータにおける予測
y_pred = model.predict(X_test_nums)
y_pred_max = np.argmax(y_pred, axis=1)  # 最尤と判断したクラスの値にする

'''
提出
'''

# 提出用データの読み込み
sub = pd.read_csv('./data/sample_submit.csv', sep=',', header=None)
print(sub.head())
    
# 目的変数カラムの置き換え
sub[1] = y_pred_max

# ダミー変数をもとの変数に戻す
sub[1] = sub[1].replace([0,1,2], ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

# ファイルのエクスポート
sub.to_csv('./submit/iris_keras.csv', header=None, index=None)

"""
スコア：
0.96
"""