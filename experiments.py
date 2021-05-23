#%%

from gym.core import ObservationWrapper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gym

import agent
import market

from gym.envs.registration import register
from gym import envs
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]

# %%

CONFIG = dict()
CONFIG['envid'] = 'market-v0'
CONFIG['entry_point'] = 'market:marketenv'


# %%

# delete if it's registered
if CONFIG['envid'] in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[CONFIG['envid']]

if CONFIG['envid'] not in env_ids:
    register(

        id = CONFIG['envid'],
        entry_point = CONFIG['entry_point']

    )

# %%


env = gym.make(CONFIG['envid'])

# %%
rewards = []; inventory = []
env.reset()
for i in range(1000):

    if i == 0 :
        ob,rew,_,information = env.step((-0.3,-0.3,0))
    else :
        ob,rew,_,information = env.step((-0.3,-0.3,inventory[-1]))

    rewards.append(rew)
    inventory.append(information['trade'])
    if i % 100 == 0 :
        print(i,';')
        print(ob)

log = pd.DataFrame(
    {'reward':rewards,
    'inventory':np.array(inventory).cumsum()}
    )

log.inventory.plot()
# %%

log.reward.cumsum().plot()
# %%
