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
y_data = np.sin(x_data)#对x_data中的元素取正弦

train_size = 2000
val_size = 200
x_train = np.zeros(2 * train_size)#4000
y_train = np.zeros(2 * train_size)
x_val = np.zeros(2 * val_size)#400
y_val = np.zeros(2 * val_size)

# Half with lower noise
train_val_idx_1 = np.random.choice(list(range(1200, 3500)), 
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
plt.plot(x_data, y_data, x_train, y_train, '.', x_val, y_val, 'o', markersize=4)#l蓝色线：原始数据及其对应sin函数值；train对应着黄色△数据，val对应着绿色圆点数据
plt.show()

train_size *=2
val_size *= 2

# ReplayBuffer generates its own training/validation split, but in this example we want to
# keep the split generated above, so instead we use two replay buffers. 

num_members = 5
train_buffer = replay_buffer.ReplayBuffer(train_size, (1,), (0,))#ReplyBuffer(capacity,obs_shape,action_shape)
val_buffer = replay_buffer.ReplayBuffer(val_size, (1,), (0,))
for i in range(train_size):
    train_buffer.add(x_train[i], 0, y_train[i], 0, False)
for i in range(val_size):
    val_buffer.add(x_val[i], 0, y_val[i], 0, False)
train_dataset, _ = train_buffer.get_iterators(#iterator:迭代器
    2048, 0, train_ensemble=True, ensemble_size=num_members, shuffle_each_epoch=True)
val_dataset, _ = train_buffer.get_iterators(2048, 0, train_ensemble=False)
#ReplayBuffer.get_iterators(batch_size, val_ratio, train_ensemble=False, ensemble_size=None, shuffle_each_epoch=True)

ensemble = models.GaussianMLP(#in_size: size of input. out_size: size of output. device: the device to use for the model
    1, 1, device, num_layers=3, hid_size=64, use_silu=True, ensemble_size=num_members)
wrapper = models.OneDTransitionRewardModel(ensemble, target_is_delta=False, normalize=True, learned_rewards=False)
#wrapper封装类

wrapper.update_normalizer(train_buffer.get_all())#Updates the normalizer statistics using the batch of transition data.
trainer = models.ModelTrainer(wrapper, optim_lr=0.003, weight_decay=5e-5)
train_losses, val_losses = trainer.train(train_dataset, val_dataset, num_epochs=500, patience=100)
fig, ax = plt.subplots(2, 1, figsize=(16, 8))
ax[0].plot(train_losses)
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("train loss (gaussian nll)")
ax[1].plot(val_losses)
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("val loss (mse)")
plt.show()

x_tensor = torch.from_numpy(x_data).unsqueeze(1).float().to(device)
x_tensor = wrapper.input_normalizer.normalize(x_tensor)#Normalizes the value according to the stored statistics

with torch.no_grad():
    y_pred, y_pred_logvar = ensemble(x_tensor)#正则化后的x放入GaussianMLP模型
    y_pred = y_pred[..., 0]
    y_pred_logvar = y_pred_logvar[..., 0]
y_var_epi = y_pred.var(dim=0).cpu().numpy()
y_var = y_pred_logvar.exp()#对y_pred_logvar中的数据取指数
y_pred = y_pred.mean(dim=0).cpu().numpy()
y_var_ale = y_var.mean(dim=0).cpu().numpy()

y_std = np.sqrt(y_var_epi + y_var_ale)
plt.figure(figsize=(16, 8))
plt.plot(x_data, y_data, 'r')
plt.plot(x_train, y_train, '.', markersize=0.9)
plt.plot(x_data, y_pred, 'b-', markersize=4)
plt.fill_between(x_data, y_pred, y_pred + 2 * y_std, color='b', alpha=0.2)
plt.fill_between(x_data, y_pred - 2 * y_std, y_pred, color='b', alpha=0.2)
plt.axis([-12, 12, -2.5, 2.5])
plt.show()