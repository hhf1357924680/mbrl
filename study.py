import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import torch.optim as optim

import mbrl.models as models
import mbrl.util.replay_buffer as replay_buffer

device = torch.device("cuda:0")

%load_ext autoreload
%autoreload 2

%matplotlib inline
mpl.rcParams['figure.facecolor'] = 'white'

x_data = np.linspace(-12, 12, 10000)#np.linspace主要用来创建等差数列，numpy.linspace(start：返回样本数据开始点, stop：返回样本数据结束点, num=50：生成的样本数据量默认)（返回的是 [start, stop]之间的均匀分布）
y_data = np.sin(x_data)

train_size = 2000
val_size = 200
x_train = np.zeros(2 * train_size)#train的样本量是4000
y_train = np.zeros(2 * train_size)
x_val = np.zeros(2 * val_size)#val的样本量是400
y_val = np.zeros(2 * val_size)

train_val_idx_1 = np.random.choice(list(range(1200, 3500)), #list(range(15,26))：[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
                                   size=train_size + val_size, #size=2200
                                   replace=False)#False表示不可以取相同数字

x_1 = np.linspace(-5, 5, 50)
y_1 = 2*(x_1)
s = 10
v = 5
x_t = np.zeros(2 * s)
y_t = np.zeros(2 * s)
x_v = np.zeros(2 * v)
y_v = np.zeros(2 * v)

m=0.5
t= np.random.choice(list(range(15,35)), #list(range(15,26))：[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
                                   size=s+v, #size=2200
                                   replace=False)
x_t[:s] = x_1[t[:s]]
y_t[:s] = y_1[t[:s]]
y_t[:s] = y_t[:s]+ m * np.random.randn(s)
x_v[:v] = x_1[t[s:]]
y_v[:v] = y_1[t[s:]] 
y_v[:v] = y_v[:v] + m * np.random.randn(v)



mag = 0.05
x_train[:train_size] = x_data[train_val_idx_1[:train_size]]
y_train[:train_size] = y_data[train_val_idx_1[:train_size]] + mag * np.random.randn(train_size)
x_val[:val_size] = x_data[train_val_idx_1[train_size:]]
y_val[:val_size] = y_data[train_val_idx_1[train_size:]] + mag * np.random.randn(val_size)

plt.figure(figsize=(5, 4))#plt.figure(figsize=(16,8))表示figure 的大小为宽、长
plt.plot(x_1, y_1, x_t, y_t, '^',x_v, y_v, 'o', markersize=6)
plt.show()


#1.从fin-rl中取出aapl的数据放入mbrl中
#2.选取2000-01-01   2021-06-20共5000左右
#3.选取1500train  500val  

df.head()
df.sort_values(["date", "tic"]).head()
#取出数据集的第1列和第3列
#方法1
df.loc[:,['close']]
data = df[['close']]
data

