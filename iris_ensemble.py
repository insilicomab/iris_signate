# -*- coding: utf-8 -*-

'''
アンサンブル学習
'''

# ライブラリのインポート
import pandas as pd
import numpy as np
from scipy import stats

# 提出用サンプルの読み込み
sub = pd.read_csv('./data/sample_submit.csv', sep=',', header=None)

# 予測データの読み込み
lgb_sub = pd.read_csv('./submit/iris_LightGBM.csv', sep=',', header=None)
xgb_sub = pd.read_csv('./submit/iris_XGboost.csv', sep=',', header=None)
lr_sub = pd.read_csv('./submit/iris_LogisticRegression.csv', sep=',', header=None)
qda_sub = pd.read_csv('./submit/iris_QDA.csv', sep=',', header=None)
bc_sub = pd.read_csv('./submit/iris_BaggingClassifier.csv', sep=',', header=None)
lda_sub = pd.read_csv('./submit/iris_LDA.csv', sep=',', header=None)
mlp_sub = pd.read_csv('./submit/iris_MLP.csv', sep=',', header=None)
rf_sub = pd.read_csv('./submit/iris_RF.csv', sep=',', header=None)

# 予測データの結合
df = pd.concat([lgb_sub[1], 
                xgb_sub[1], 
                lr_sub[1], 
                qda_sub[1], 
                bc_sub[1], 
                lda_sub[1],
                mlp_sub[1],
                rf_sub[1]
                ],
               axis=1)

# ダミー変数化
df = df.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0,1,2])

# アンサンブル学習
ensemble_array = np.array(df).T
pred = stats.mode(ensemble_array)[0].T # 予測データリストのうち最頻値を算出し、行と列を入れ替え

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
sub.to_csv('./submit/iris_ensemble.csv', header=None, index=None)