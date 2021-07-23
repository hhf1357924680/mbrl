
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


mpl.rcParams['figure.facecolor'] = 'white'

x_data = np.array(range(5401))
y_data=pd.read_csv("/mnt/work/project/mbrl-lib/dataset/mbrl.csv",usecols=[5])
y_data = np.array(list(y_data["close"]))


train_size = 1000
val_size = 100
x_train = np.zeros(2 * train_size)
y_train = np.zeros(2 * train_size)
x_val = np.zeros(2 * val_size)
y_val = np.zeros(2 * val_size)

# Half with lower noise
train_val_idx_1 = np.random.choice(list(range(0, 2499)), 
                                   size=train_size + val_size, 
                                   replace=False)


mag = 0.0
x_train[:train_size] = x_data[train_val_idx_1[:train_size]]
y_train[:train_size] = y_data[train_val_idx_1[:train_size]] + mag * np.random.randn(train_size)
y_val[:val_size] = y_data[train_val_idx_1[train_size:]] + mag * np.random.randn(val_size)

# Half with higher noise
train_val_idx_2 = np.random.choice(list(range(2500, 5401)), 
                                   size=train_size + val_size, 
                                   replace=False)
mag = 0.0
x_train[train_size:] = x_data[train_val_idx_2[:train_size]]
y_train[train_size:] = y_data[train_val_idx_2[:train_size]] + mag * np.random.randn(train_size)
x_val[val_size:] = x_data[train_val_idx_2[train_size:]]
y_val[val_size:] = y_data[train_val_idx_2[train_size:]] + mag * np.random.randn(val_size)

plt.figure(figsize=(16, 8))
plt.plot(x_data, y_data, x_train, y_train, '^', x_val, y_val, 'o', markersize=4)#l蓝色线：原始数据及其对应sin函数值；train对应着黄色△数据，val对应着绿色圆点数据
plt.show()

train_size *=2
val_size *= 2

# ReplayBuffer generates its own training/validation split, but in this example we want to
# keep the split generated above, so instead we use two replay buffers. 

num_members = 5
train_buffer = replay_buffer.ReplayBuffer(train_size, (1,), (0,))
val_buffer = replay_buffer.ReplayBuffer(val_size, (1,), (0,))
for i in range(train_size):
    train_buffer.add(x_train[i], 0, y_train[i], 0, False)
for i in range(val_size):
    val_buffer.add(x_val[i], 0, y_val[i], 0, False)
train_dataset, _ = train_buffer.get_iterators(
32, 0, train_ensemble=True, ensemble_size=num_members, shuffle_each_epoch=True)
val_dataset, _ = train_buffer.get_iterators(32, 0, train_ensemble=False)



import pathlib
import warnings
from typing import List, Optional, Sequence, Sized, Tuple, Union
ensemble = models.GaussianMLP(
    1, 1, device, num_layers=3, hid_size=64, use_silu=True, ensemble_size=num_members)
wrapper = models.OneDTransitionRewardModel(ensemble, target_is_delta=False, normalize=True, learned_rewards=False)

wrapper.update_normalizer(train_buffer.get_all())
trainer = models.ModelTrainer(wrapper, optim_lr=0.003, weight_decay=5e-5)
train_losses, val_losses = trainer.train(train_dataset, val_dataset, num_epochs=500, patience=100)


import numpy as np

from mbrl.types import TransitionBatch


def _consolidate_batches(batches: Sequence[TransitionBatch]) -> TransitionBatch:
    len_batches = len(batches)
    b0 = batches[0]
    obs = np.empty((len_batches,) + b0.obs.shape, dtype=b0.obs.dtype)#Return a new array of given shape and type, without initializing entries.
    act = np.empty((len_batches,) + b0.act.shape, dtype=b0.act.dtype)
    next_obs = np.empty((len_batches,) + b0.obs.shape, dtype=b0.obs.dtype)
    rewards = np.empty((len_batches,) + b0.rewards.shape, dtype=np.float32)
    dones = np.empty((len_batches,) + b0.dones.shape, dtype=bool)
    for i, b in enumerate(batches):
        obs[i] = b.obs
        act[i] = b.act
        next_obs[i] = b.next_obs
        rewards[i] = b.rewards
        dones[i] = b.dones
    return TransitionBatch(obs, act, next_obs, rewards, dones)

git@github.com:hhf1357924680/mbrl.git