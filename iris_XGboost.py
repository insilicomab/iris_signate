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

# ライブラリのインポート
from sklearn.preprocessing import LabelEncoder

# object型の変数の取得
categories = train.columns[train.dtypes == 'object']
print(categories)

# 'class'のダミー変数化
le = LabelEncoder()
le = le.fit(train['class'])
train['class'] = le.transform(train['class'])

'''
モデルの構築と評価
'''

# ライブラリのインポート
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from statistics import mean

# 5分割する
folds = 5
skf = StratifiedKFold(n_splits=folds)

# ハイパーパラメータの設定
params = {
    # 多値分類問題
    'objective': 'multi:softmax',
    # クラス数は 3
    'num_class': 3
}

# 説明変数と目的変数を指定
X_train = train.drop(['class', 'id'], axis=1)
Y_train = train['class']

# 各foldごとに作成したモデルごとの予測値を保存
models = []
scores = []
oof = np.zeros(len(X_train))

for train_index, val_index in skf.split(X_train, Y_train):
    x_train = X_train.iloc[train_index]
    x_valid = X_train.iloc[val_index]
    y_train = Y_train.iloc[train_index]
    y_valid = Y_train.iloc[val_index]
    
    xgb_train = xgb.DMatrix(x_train, label=y_train)
    xgb_eval = xgb.DMatrix(x_valid, label=y_valid)   
    evals = [(xgb_train, "train"), (xgb_eval, "eval")]
    model = xgb.train(params, xgb_train,
                      evals=evals,
                      num_boost_round=1000,
                      early_stopping_rounds=20,
                      verbose_eval=20)
    
    y_pred = model.predict(xgb_eval)
    score = accuracy_score(y_valid, y_pred)
    print(score)
    
    models.append(model)
    scores.append(score)
    oof[val_index] = y_pred
    
    # 混同行列の作成
    cm = confusion_matrix(y_valid, y_pred)
    
    # heatmapによる混同行列の可視化
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()
    

# 平均accuracy scoreを計算する
print(mean(scores))

"""
予測精度：
0.92
"""

'''
テストデータの予測
'''

# ライブラリのインポート
from scipy import stats

# 説明変数と目的変数を指定
X_test = test.drop('id', axis=1)
xgb_test = xgb.DMatrix(X_test)

# テストデータにおける予測
preds = []

for model in models:
    y_pred_test = model.predict(xgb_test)
    preds.append(y_pred_test)
    
# アンサンブル学習
preds_array = np.array(preds)
pred = stats.mode(preds_array)[0].T # 予測データリストのうち最頻値を算出し、行と列を入れ替え

'''
提出
'''

# 提出用データの読み込み
sub = pd.read_csv('./data/sample_submit.csv', sep=',', header=None)
print(sub.head())
    
# 目的変数カラムの置き換え
sub[1] = pred

# ダミー変数をもとの変数に戻す
sub[1] = sub[1].replace([0,1,2], ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

# ファイルのエクスポート
sub.to_csv('./submit/iris_XGboost.csv', header=None, index=None)

"""
スコア：
0.9600000
"""