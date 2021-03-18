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
ネットワークの定義
'''

# ライブラリのインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# classとしてネットワークを作成
class IrisNet(nn.Module):
    
    # init関数の定義
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8, 3)
    
    # forward関数の定義
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

'''
モデルの構築と評価
'''

# ライブラリのインポート
from sklearn.model_selection import StratifiedKFold

#  定数定義
folds = 3
skf = StratifiedKFold(n_splits=folds)
epoch = 5

# 説明変数と目的変数を指定
X_train = train.drop(['class', 'id'], axis=1)
Y_train = train['class']

#  K分割交差検証
for train_index, val_index in skf.split(X_train, Y_train):
    
    x_train = np.array(X_train.iloc[train_index]).astype(np.float32)
    x_valid = np.array(X_train.iloc[val_index]).astype(np.float32)
    y_train = np.array(Y_train.iloc[train_index]).astype(np.float32)
    y_valid = np.array(Y_train.iloc[val_index]).astype(np.float32)
    
    x = torch.tensor(x_train,dtype = torch.float)
    y = torch.tensor(y_train)
    net = IrisNet()
    
    # optimezer定義
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    
    # loss関数定義
    criterion = nn.CrossEntropyLoss()
    
    for i in range(epoch):
        # 勾配初期化
        optimizer.zero_grad()
        output = net(x)
        
        # loss算出
        loss = criterion(output, y.type(torch.long)) # yをLongタイプに変換
        loss.backward()
        optimizer.step()
        print('epoch: {},'.format(i) + 'loss: {:.10f}'.format(loss))
    
    
    # 精度計算
    outputs = net(torch.tensor(x_valid, dtype = torch.float))
    _, y_pred = torch.max(outputs.data, 1)
    accuracy = 100 * np.sum(y_pred.numpy() == y_valid) / len(y_pred)
    print('accuracy: {:.1f}%'.format(accuracy))



"""
予測精度：
0.9466666666666667
"""