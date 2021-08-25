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
                                                    random_state=0)

# モデル構造の定義
Dense = keras.layers.Dense

from pylab import rcParams
plt.figure(figsize=(10,6),dpi=200)
plt.rcParams['figure.figsize'] = (10 ,6)

def fit(epochs):
    # モデルの構造を定義
    model = keras.models.Sequential()
    model.add(Dense(10, activation='relu', input_shape=(4,)))
    model.add(Dense(3, activation='softmax'))
    # モデルを構築
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    # 学習を実行
    result = model.fit(x_train, y_train,
        batch_size=20,
        epochs=epochs)
    # モデルを評価
    score = model.evaluate(x_valid, y_valid, verbose=1)
    print('正解率=', score[1], 'loss=', score[0])
    # グラフを描画
    plt.plot(range(1, epochs+1), result.history['accuracy'], label="training")
    plt.plot(range(1, epochs+1), result.history['loss'], label="loss")
    plt.xlabel('Epochs=' + str(epochs))
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
fit(1000)