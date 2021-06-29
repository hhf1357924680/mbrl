#Fin - Gaussian
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
import pandas as pd
import torch.optim as optim

import mbrl.models as models
import mbrl.util.replay_buffer as replay_buffer

device = torch.device("cuda:0")

%load_ext autoreload
%autoreload 2

%matplotlib inline
mpl.rcParams['figure.facecolor'] = 'white'

#读取AAPL的close数据
import csv
close=pd.read_csv("/mnt/work/project/mbrl-lib/dataset/mbrl.csv",usecols=[5])
close.to_csv("/mnt/work/project/mbrl-lib/dataset/close.csv")
close#2000-1-1  2021-6-20的所有close数据，共5401条
x0_data = close[:3000]
x_data = close[3000:]
 
import matplotlib.pyplot as plt
x_data.plot()
plt.grid(True)
plt.show()

train_size = 300
x_train = np.zeros(2 * train_size)

# Half with lower noise
mag = 0.05
x_train = x_data[4200:4200+train_size] 
x_train = np.array(x_train)
x_train = x_train + mag * np.random.randn(train_size)
x_train

plt.figure(figsize=(16, 8))#plt.figure(figsize=(16,8))表示figure 的大小为宽、长
#plt.plot(range(len(x_data)), x_data, '.', range(len(x_train)), x_train, 'o', markersize=2)
plt.plot( x0_data, '.', x_data,'o',  markersize=2)
plt.close()

plt.legend()
plt.xlabel('date')
plt.ylabel('close prise')
plt.show()

train_size *=2

# ReplayBuffer generates its own training/validation split, but in this example we want to
# keep the split generated above, so instead we use two replay buffers. 

num_members = 5
train_buffer = replay_buffer.ReplayBuffer(train_size, (1,), (0,))#ReplyBuffer(capacity,obs_shape,action_shape)
#val_buffer = replay_buffer.ReplayBuffer(val_size, (1,), (0,))
for i in range(len(x0_data)): ## old ! range(train_size):
    train_buffer.add(i, 0, x0_data.iloc[i], 0, False)
for i in range(len(x_data)): ## old ! range(train_size):
    train_buffer_1.add(i, 0, x_data.iloc[i], 0, False)

   ##old! train_buffer.add(x_train[i], False)
#for i in range(val_size):
#    val_buffer.add(x_val[i], 0, y_val[i], 0, False)
train_dataset, _ = train_buffer.get_iterators(#iterator:迭代器
    2048, 0, train_ensemble=True, ensemble_size=num_members, shuffle_each_epoch=True)
#val_dataset, _ = train_buffer_1.get_iterators(2048, 0, train_ensemble=False)

#ReplayBuffer.get_iterators(batch_size, val_ratio, train_ensemble=False, ensemble_size=None, shuffle_each_epoch=True)

ensemble = models.GaussianMLP(#in_size: size of input. out_size: size of output. device: the device to use for the model
    1, 1, device, num_layers=3, hid_size=64, use_silu=True, ensemble_size=num_members)
wrapper = models.OneDTransitionRewardModel(ensemble, target_is_delta=False, normalize=True, learned_rewards=False)
#wrapper封装类

wrapper.update_normalizer(train_buffer.get_all())#Updates the normalizer statistics using the batch of transition data.
trainer = models.ModelTrainer(wrapper, optim_lr=0.003, weight_decay=5e-5)
train_losses = trainer.train(train_dataset, num_epochs=500, patience=100)


fig, ax = plt.subplots(2, 1, figsize=(16, 8))
ax[0].plot(train_losses[0])
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("train loss (gaussian nll)")
plt.show()


x_data = np.array(range(len(x0_data)))
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
plt.plot(x_data, 'r')
plt.plot(x_train,'.', markersize=0.9)
plt.plot(x_data, y_pred, 'b-', markersize=4)
plt.fill_between(x_data, y_pred, y_pred + 2 * y_std, color='b', alpha=0.2)
plt.fill_between(x_data, y_pred - 2 * y_std, y_pred, color='b', alpha=0.2)
plt.axis([-12, 12, -2.5, 2.5])
plt.show()