#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
# import shutil
# import collections

import gym
from gym import wrappers
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models import PolicyNetwork, CriticNetwork

from gym.envs.registration import register
from gym import envs

all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs ]

# %%
CONFIG = dict()
CONFIG['envid'] = 'market-v0'
CONFIG['entry_point'] = 'market:marketenv'

# %%

# delete if it's registered
if CONFIG['envid'] in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[CONFIG['envid']]

if CONFIG['envid'] not in gym.envs.registry.env_specs:
    register(

        id = CONFIG['envid'],
        entry_point = CONFIG['entry_point']

    )


#%%

class envlogger:

    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIPRANGE = 0.2
    OPT_ITER = 20
    BATCH_SIZE = 2048

    def __init__(self,env_id,max_timesteps,action_space=3,
        trajectory_size=1024
    ):

        self.env = gym.make(env_id)
        self.max_timesteps = max_timesteps

        self.policy = PolicyNetwork(action_space=action_space)
        self.old_policy = PolicyNetwork(action_space=action_space)
        self.critic = CriticNetwork()

        self.iter_traj = 0 
        self.trajectory_size = trajectory_size

        self.policy(np.atleast_2d(self.env.generate_initial_states()))
        self.old_policy(np.atleast_2d(self.env.generate_initial_states()))

    def reset(self):
        self.env.reset()

    def reset_env(self):
        self.env.reset()

    def step(self,action):

        self.iter_traj += 1

        observation, reward, done, info = self.env.step(action)
        self.trajectory['s'].append(
            np.append(observation['bid_s'],observation['ask_s'])
        )

        self.trajectory['a'].append(action)
        self.trajectory['r'].append(reward)
        self.trajectory['t'].append(info['trade'])

        if len(self.trajectory['t']) > 100 :
            self.trajectory['state'].append( np.append(
                np.append(observation['bid_s'],observation['ask_s']),
                np.array(self.trajectory['t'][-100:],dtype=np.float32)
            ))
            return(self.trajectory['state'][-1],info['trade'])
        else:
            self.trajectory['state'].append( np.append(
                np.append(observation['bid_s'],observation['ask_s']),
                np.append(np.zeros(100-len(self.trajectory['t'])),
                    np.array(self.trajectory['t'],dtype=np.float32)
                )
            ))
            return(self.trajectory['state'][-1],info['trade'])

    def run(self,n_updates,logdir=None):

        history = {"epoch":[],"score":[]}

        for epoch in range(n_updates):

            self.env.reset()
            state = self.env.generate_initial_states()

            for _ in range(self.trajectory_size+100):
                action = self.policy.sample_action(state)
                next_state,reward,done,info = self.env.step(np.append(action))
                state = next_state

            trajectory = self.env.get_trajectory()
            trajectory = self.compute_advantage(trajectory,state)

            vloss = self.update_critic(
                trajectory['state'][100:],
                trajectory['R'][100:]
            )

            self.update_policy(
                trajectory['state'][100:],
                trajectory['a'][100:,:],
                trajectory['advantage'][100:]
            )

            score = np.array(trajectory['r']).sum()

            history['epoch'].append(epoch)
            history['score'].append(score)

            if epoch % 5 == 0 :
                print(str(epoch),':')
                pd.DataFrame(history).plot(x='epoch',y='score')

        return(pd.DataFrame(history))


    def compute_advantage(self,trajectory:dict,state_last_plus_one:np.ndarray):

        '''
            generalized advantage estimation (gae,2016)
        '''

        trajectory['v_pred'] = self.critic(trajectory['state']).numpy(
        ).reshape(-1)
        trajectory['v_pred_next'] = np.append(
            trajectory['v_pred'][1:], 
            self.critic(np.atleast_2d(state_last_plus_one)).numpy()
        )
        

        normed_rewards = trajectory['r']
        deltas = normed_rewards+self.GAMMA * trajectory['v_pred_next'] \
            - trajectory['v_pred']

        advantages = np.zeros_like(deltas,dtype=np.float32)

        lastgae = 0 
        for i in reversed(range(len(deltas))):

            lastgae = deltas[i] + self.GAMMA * self.GAE_LAMBDA * lastgae
            advantages[i] = lastgae

        trajectory['advantage'] = advantages
        trajectory['R'] = advantages + trajectory['v_pred']

        return trajectory

    def update_policy(self,states,actions,advantages):

        self.old_policy.set_weights(self.policy.get_weights())
        indices = np.random.choice(
            range(states.shape[0]),(self.OPT_ITER,self.BATCH_SIZE)
        )

        for i in range(self.OPT_ITER):
            idx = indices[i]
            old_means, old_stdevs = self.old_policy(states[idx])
            old_logprob = self.compute_logprob(
                old_means,old_stdevs,actions[idx]
            )

            with tf.GradientTape() as tape :

                new_means, new_stdevs = self.policy(states[idx])
                new_logprob = self.compute_logprob(
                    new_means,new_stdevs,actions[idx]
                )
                ratio = tf.exp(new_logprob-old_logprob)
                ratio_clipped = tf.clip_by_value(
                    ratio,1-self.CLIPRANGE,1+self.CLIPRANGE
                )
                loss_unclipped = ratio*advantages[idx]
                loss_clipped = ratio_clipped*advantages[idx]
                loss = tf.minimum(loss_unclipped,loss_clipped)
                loss = -1* tf.reduce_mean(loss)

            grads = tape.gradient(loss,self.policy.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads,0.5)
            self.policy.optimizer.apply_gradients(
                zip(grads,self.policy.trainable_variables)
            )


    def update_critic(self,states,v_targs):

        losses = []
        indices = np.random.choice(
            range(states.shape[0]),(self.OPT_ITER,self.BATCH_SIZE)
        )
        

        for i in range(self.OPT_ITER):

            idx = indices[i]
            old_vpred = self.critic(states[idx])

            with tf.GradientTape() as tape:

                vpred = self.critic(states[idx])
                vpred_clipped = old_vpred + tf.clip_by_value(
                    vpred - old_vpred, -self.CLIPRANGE,self.CLIPRANGE
                )
                loss = tf.maximum(
                    tf.square(v_targs[idx]-vpred),
                    tf.square(v_targs[idx]-vpred_clipped)
                )
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss,self.critic.trainable_variables)
            grads,_ = tf.clip_by_global_norm(grads,0.5)
            self.critic.optimizer.apply_gradients(
                zip(grads,self.critic.trainable_variables)
            )
            losses.append(loss)

        return np.array(losses).mean()


    @tf.function
    def compute_logprob(self,means,stdevs,actions):

        logprob = -0.5 * np.log(2*np.pi)
        logprob += -tf.math.log(stdevs)
        logprob += -0.5 * tf.square((actions-means)/stdevs)
        logprob = tf.reduce_sum(logprob,axis=1,keepdims=True)

        return(logprob)




# %%

mkt = envlogger(CONFIG['envid'],1000)
mkt.reset()
inventory = 0 
output = mkt.run(n_updates=100)


# %%


