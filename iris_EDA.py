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

# データの確認
print(train.head())
print(train.dtypes)

'''
sepalとpetalの面積と長さ／幅の特徴量追加
'''

# ライブラリのインポート
from sklearn.preprocessing import StandardScaler

# sepalとpetalの面積
train['sepal_area'] = train['sepal length in cm'] * train['sepal width in cm']
train['petal_area'] = train['petal length in cm'] * train['petal width in cm']

# length/widthの割合
train['sepal_length/width'] = train['sepal length in cm'] / train['sepal width in cm']
train['petal_length/width'] = train['petal length in cm'] / train['petal width in cm']

# 説明変数を指定
X_train = train.drop(['class', 
                      'id', 
                      'sepal length in cm',
                      'sepal width in cm',
                      'petal length in cm',
                      'petal width in cm'], axis=1)
Y_train = train['class']

# 説明変数の標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

# 標準化した説明変数をDataFrame化する
X_train_std = pd.DataFrame(X_train_std)
X_train_std.columns = X_train.columns # カラム名を元に戻す

# 標準化した説明変数と目的変数の再結合
train_std = pd.concat([X_train_std, Y_train], axis=1)

# sepalの面積のプロット
fig, ax = plt.subplots(1, 2, figsize=(10,5))
sns.stripplot(x='class', y='sepal_area', data=train, ax=ax[0])
sns.stripplot(x='class', y='sepal_area', data=train_std, ax=ax[1])
ax[0].set_title("sepal area")
ax[1].set_title("sepal area(standarized)")
plt.show()

# petalの面積のプロット
fig, ax = plt.subplots(1, 2, figsize=(10,5))
sns.stripplot(x='class', y='petal_area', data=train, ax=ax[0])
sns.stripplot(x='class', y='petal_area', data=train_std, ax=ax[1])
ax[0].set_title("petal area")
ax[1].set_title("petal area(standarized)")
plt.show()

# sepalのlength/widthの割合のプロット
fig, ax = plt.subplots(1, 2, figsize=(10,5))
sns.stripplot(x='class', y='sepal_length/width', data=train, ax=ax[0])
sns.stripplot(x='class', y='sepal_length/width', data=train_std, ax=ax[1])
ax[0].set_title("sepal length/width")
ax[1].set_title("sepal length/width(standarized)")
plt.show()

# petalのlength/widthの割合のプロット
fig, ax = plt.subplots(1, 2, figsize=(10,5))
sns.stripplot(x='class', y='petal_length/width', data=train, ax=ax[0])
sns.stripplot(x='class', y='petal_length/width', data=train_std, ax=ax[1])
ax[0].set_title("petal length/width")
ax[1].set_title("petal length/width(standarized)")
plt.show()