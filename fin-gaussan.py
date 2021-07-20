import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import torch.optim as optim
import pandas as pd

import mbrl.models as models
import mbrl.util.replay_buffer as replay_buffer

device = torch.device("cuda:0")

#x_data = np.linspace(-12, 12, 10000)#np.linspace主要用来创建等差数列，numpy.linspace
x_data = np.array(range(5400))
y_data=pd.read_csv("/mnt/work/project/mbrl-lib/dataset/mbrl.csv",usecols=[5])
y_data = np.array(list(y_data["close"]))
#y_data = np.sin(x_data)#对x_data中的元素取正弦

train_size = 1000
val_size = 100
x_train = np.zeros(2 * train_size)#4000
y_train = np.zeros(2 * train_size)
x_val = np.zeros(2 * val_size)#400
y_val = np.zeros(2 * val_size)

# Half with lower noise
train_val_idx_1 = np.random.choice(list(range(0, 5400)), 
                                   size=train_size + val_size, 
                                   replace=False)
#numpy.random.choice(a, size=None, replace=True, p=None);
#从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组,此处a是list(range(15,20))：[15, 16, 17, 18, 19];
#replace:True表示可以取相同数字，False表示不可以取相同数字;
#数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

mag = 0.05
x_train[:train_size] = x_data[train_val_idx_1[:train_size]]#先从train_val_idx_1中选出前train_size个位置数，对应每个位置数提取出x_data中的相应位置的数据（从0数起）；
y_train[:train_size] = y_data[train_val_idx_1[:train_size]] + mag * np.random.randn(train_size)#np.random.randn随机生成train_size个0-1之间的数
x_val[:val_size] = x_data[train_val_idx_1[train_size:]]
y_val[:val_size] = y_data[train_val_idx_1[train_size:]] + mag * np.random.randn(val_size)

# Half with higher noise
train_val_idx_2 = np.random.choice(list(range(6500, 8800)), 
                                   size=train_size + val_size, 
                                   replace=False)
mag = 0.20
x_train[train_size:] = x_data[train_val_idx_2[:train_size]]
y_train[train_size:] = y_data[train_val_idx_2[:train_size]] + mag * np.random.randn(train_size)
x_val[val_size:] = x_data[train_val_idx_2[train_size:]]
y_val[val_size:] = y_data[train_val_idx_2[train_size:]] + mag * np.random.randn(val_size)

plt.figure(figsize=(16, 8))#plt.figure(figsize=(16,8))表示figure 的大小为宽、长
plt.plot(x_data, y_data, x_train, y_train, '^', x_val, y_val, 'o', markersize=4)#l蓝色线：原始数据及其对应sin函数值；train对应着黄色△数据，val对应着绿色圆点数据
plt.show()

train_size *=2
val_size *= 2