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
from sklearn.preprocessing import StandardScaler

# object型の変数の取得
categories = train.columns[train.dtypes == 'object']
print(categories)

# 'class'のダミー変数化
le = LabelEncoder()
le = le.fit(train['class'])
train['class'] = le.transform(train['class'])

# 説明変数と目的変数を指定
X_train = train.drop(['class', 'id'], axis=1)
Y_train = train['class']

# 説明変数の標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

'''
モデルの構築と評価
'''

# ライブラリのインポート
from sklearn.utils import all_estimators
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# 5分割する
folds = 5
skf = StratifiedKFold(n_splits=folds)

# モデリングと
allAlgorithms = all_estimators(type_filter="classifier")

for(name, algorithm) in allAlgorithms:
    
    if name == 'ClassifierChain':
        continue
    
    elif name == 'MultiOutputClassifier':
        continue
    
    elif name == 'OneVsOneClassifier':
        continue
    
    elif name == 'OneVsRestClassifier':
        continue
    
    elif name == 'OutputCodeClassifier':
        continue
    
    elif name == 'StackingClassifier':
        continue
    
    elif name == 'VotingClassifier':
        continue
    
    else:
        clf = algorithm()
        try:  # Errorがでるものがあるので、try文を入れる
            if hasattr(clf,"score"):
            # クロスバリデーション
                scores = cross_val_score(clf, X_train_std, Y_train, cv=skf)
                print(f"{name:<35}の正解率= {np.mean(scores)}")
    
        except:
            pass


"""
予測精度：
AdaBoostClassifier            の正解率= 0.9333333333333333
BaggingClassifier             の正解率= 0.9600000000000002
BernoulliNB                   の正解率= 0.8133333333333332
CalibratedClassifierCV        の正解率= 0.8
CategoricalNB                 の正解率= nan
ComplementNB                  の正解率= nan
DecisionTreeClassifier        の正解率= 0.9199999999999999
DummyClassifier               の正解率= 0.3866666666666666
ExtraTreeClassifier           の正解率= 0.96
ExtraTreesClassifier          の正解率= 0.9466666666666667
GaussianNB                    の正解率= 0.9333333333333332
GaussianProcessClassifier     の正解率= 0.9199999999999999
GradientBoostingClassifier    の正解率= 0.9733333333333334
HistGradientBoostingClassifierの正解率= 0.96
KNeighborsClassifier          の正解率= 0.9199999999999999
LabelPropagation              の正解率= 0.9466666666666667
LabelSpreading                の正解率= 0.9466666666666667
LinearDiscriminantAnalysis    の正解率= 0.9866666666666667
LinearSVC                     の正解率= 0.9333333333333332
LogisticRegression            の正解率= 0.9866666666666667
LogisticRegressionCV          の正解率= 0.9733333333333334
MLPClassifier                 の正解率= 0.9600000000000002
MultinomialNB                 の正解率= nan
NearestCentroid               の正解率= 0.8666666666666668
NuSVC                         の正解率= 0.9333333333333333
PassiveAggressiveClassifier   の正解率= 0.8800000000000001
Perceptron                    の正解率= 0.8266666666666665
QuadraticDiscriminantAnalysis の正解率= 0.9866666666666667
RadiusNeighborsClassifier     の正解率= nan
RandomForestClassifier        の正解率= 0.9333333333333332
RidgeClassifier               の正解率= 0.7866666666666665
RidgeClassifierCV             の正解率= 0.8
SGDClassifier                 の正解率= 0.9066666666666668
SVC                           の正解率= 0.9333333333333333
"""

