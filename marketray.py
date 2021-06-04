# %%


import numpy as np
import pandas as pd
from scipy.stats import gamma
import agent
import gym

agenet_dict = {
    'random':agent.RandomAgent

}

# %%


class stock:

    def __init__(self,mu=0,sigma=.2,seed=None):

        self.mu = mu
        self.sigma = sigma
        self.price = 1.
        self.price_old = 1.
        self.ask_price = np.ones(10)
        self.bid_price = np.ones(10)
        self.ask_v = np.ones(10)
        self.bid_v = np.ones(10)
        self.ask_vcum = np.ones(10)
        self.bid_vcum = np.ones(10)
        self.ref_spread_curve_bid = np.ones(20)
        self.ref_spread_curve_ask = np.ones(20)
        self.price_log = [self.price]
        self.seed = seed

    def reset(self):

        self.price = 1.
        self.price_old = 1.
        self.ask_price = np.ones(10)
        self.bid_price = np.ones(10)
        self.ask_v = np.ones(10)
        self.bid_v = np.ones(10)
        self.ask_vcum = np.ones(10)
        self.bid_vcum = np.ones(10)
        self.ref_spread_curve_bid = np.ones(20)
        self.ref_spread_curve_ask = np.ones(20)
        self.price_log = [self.price]

        self.price_drift()
        self.set_lob()
        self.convert_to_spread_curve()

    def price_drift(self,delta=15./60./24./365.):

        self.price_old = self.price

        self.price = self.price * np.exp(
            self.mu*delta - 0.5*self.sigma**2*delta \
            + self.sigma*np.sqrt(delta)*np.random.randn()
        )
        
        # self.price_log.append(self.price)

    def set_lob(self,lam=1.,shape_coef=[-0.01555,0.062222,1.],seed=None):
        """
            set limit order board

            a(x-2)^2+b
            4a+b=1
            49a+b=0.3

        """

        # Spread and Price
        s0 = 0.0001 + 0.00005*np.random.randn()
        s0 = np.clip(s0,0.00001,0.00025)

        self.ask_price = self.price * ( np.ones(10) + \
            (s0/2.+np.arange(10)*0.0001) )
        self.bid_price = self.price * ( np.ones(10) - \
            (s0/2.+np.arange(10)*0.0001) )

        # Volume
        if seed is not None:
            v0 = gamma.rvs(100.,scale=0.01,random_state=seed)
        else :
            v0 = gamma.rvs(100.,scale=0.01)

        shape = np.array(
            [shape_coef[0]*(n**2)+shape_coef[1]*n+shape_coef[2] \
                for n in range(1,10)]
            )
        shape = np.append(np.array([1.]),shape)

        self.ask_v = lam * v0 * shape 
        self.bid_v = lam * v0 * shape 

        self.ask_vcum = self.ask_v.cumsum()
        self.bid_vcum = self.bid_v.cumsum()
    

    def convert_to_spread_curve(self):

        def lastsmaller(lis:np.array,x:int) -> int :
            return( min([int((lis < x).sum()),len(lis)-1] ) )

        a=np.zeros(20);b=np.zeros(20)
        for i in range(20):
            idx = lastsmaller(self.ask_vcum,i+1)
            a[i] = (self.ask_price[:(idx+1)] * self.ask_v[:(idx+1)] 
                    / self.ask_vcum[idx] ).sum()
            idx = lastsmaller(self.bid_vcum,i+1)
            b[i] = (self.bid_price[:(idx+1)] * self.bid_v[:(idx+1)] 
                    / self.bid_vcum[idx] ).sum()
            

        a = a - self.price
        b = self.price - b 

        self.ref_spread_curve_ask = a
        self.ref_spread_curve_bid = b





# %%

class investor:

    def __init__(self):
        self.order = 0.

    def generate_order(self):
        self.order = np.random.choice([-1,1],1)
        return(self.order)


class investors:

    def __init__(self,ninvestors=20):

        self.ninvestors=ninvestors
        self.investors = [ investor() for _ in range(ninvestors) ]
        self.orders = np.zeros(len(self.investors))
        self.sell = 0
        self.buy = 0 

    def reset(self):

        self.generate_orders()

    def generate_orders(self):

        for i in range(len(self.investors)):
            self.orders[i] = self.investors[i].generate_order()

        sell = 0 ; buy = 0 
        for o in self.orders :
            if o > 0 :
                buy+=1
            else :
                sell+=1
        
        self.sell=sell
        self.buy = buy

# %%


class marketenv(gym.Env):

    def __init__(self,env_config):

        # max_iter=10000,competitor_type:str='random',
        # ninvestor:int=20, agent_config:dict = {'random_space':[-1.,1.]}
        # , asset_config:dict={'mu':0.0,'sigma':0.10}

        self.max_iter = env_config['max_iter']
        self.iter = 0 
        self.done = False

        self.asset = stock(
            mu=env_config['asset_config']['mu'],
            sigma=env_config['asset_config']['sigma']
            )
        self.invstrs = investors()

        self.inventory = 0
        self.inventory_old = 0
        self.hedge = 0 
        self.hedge_old = 0

        if env_config['competitor_type'] == "random" :
            self.competitor = agenet_dict[
                env_config['competitor_type']
                ](
                    env_config['agent_config']['random_space']
                )

        self.trajectory = {"s":[],"a":[],"i":[],"h":[],
            "t":[],"r":[],'state':[],'spl':[],'ipl':[],'hc':[]}


    def step(self,action:tuple):

        '''

            Parameters
            __________
            action: tuple
                (bid_spread, ask_spread, hedge)

            Returns
            _______
            tuple
                observations: np.array
                    trade executed in previous time step 
                        np.array(n_investors)
                    inventory: int
                        for the time step t
                    refereence price :float
                    reference spread curve: np.array
                    InventoryPL: float
                rewards:float
                done:Boolen
                info:tuple
        '''

        # calc reward
        reward=[0,0,0,0]
        traded = [0,0]
        orders = np.zeros(len(self.invstrs.orders))

        # log interaction with each investors
        if self.competitor.eps_bid > action[0]:
            for i in range(len(self.invstrs.orders)):
                if self.invstrs.orders[i] == -1:
                    orders[i] = 1
        if self.competitor.eps_ask > action[1]:
            for i in range(len(self.invstrs.orders)):
                if self.invstrs.orders[i] == 1:
                    orders[i] = -1

        # new hedge cost
        new_hedge = int(self.inventory*action[2]) - \
            int(self.inventory_old * self.hedge ) 
        if new_hedge > 0 :
            new_hedge_ = min(new_hedge,len(self.asset.ref_spread_curve_ask))
            hc = self.asset.ref_spread_curve_ask[new_hedge_-1]
        elif new_hedge <= 0 :
            new_hedge_ = min(-new_hedge,len(self.asset.ref_spread_curve_bid))
            hc = self.asset.ref_spread_curve_bid[new_hedge_-1]
        reward[3] = -1. * hc * np.abs(new_hedge)

        # update hedge position
        self.hedge_old = self.hedge
        self.hedge = action[2]

        # Spread PL
        #bid
        if (self.competitor.eps_bid > action[0]) & (self.invstrs.sell > 0) :
            reward[0] = self.asset.ref_spread_curve_bid[
                self.invstrs.sell - 1
            ] * (1.+action[0]) * self.invstrs.sell
            traded[0]=self.invstrs.sell # investor sell agent buy
        #ask
        if (self.competitor.eps_ask > action[1]) & (self.invstrs.buy > 0 ):
            reward[1] = self.asset.ref_spread_curve_ask[
                self.invstrs.buy - 1
            ] * (1.+action[1]) * self.invstrs.buy
            traded[1]=-self.invstrs.buy # investor buy agent sell

        # update stock
        self.asset.price_drift()
        self.asset.set_lob(lam=5.)
        self.asset.convert_to_spread_curve()

        # update investors
        self.invstrs.generate_orders()

        # udpate competitor
        self.competitor.set_offer_spread()

        # inventory PL
        #this is defined z_t(P_{t+1}-P_{t})
        reward[2] = self.inventory * (1. - self.hedge) * \
            (self.asset.price - self.asset.price_old)

        # new state
        self.invenotry_old = self.inventory
        self.inventory += np.sum(traded)
        spcurve = np.append(self.asset.ref_spread_curve_bid,
                self.asset.ref_spread_curve_ask
            )
        self.trajectory['s'].append(spcurve)
        self.trajectory['t'].append(np.array(traded).sum())
        self.trajectory['i'].append(self.inventory)
        self.trajectory['h'].append(self.hedge)
        self.trajectory['a'].append(action)
        self.trajectory['r'].append(np.sum(reward))
        self.trajectory['spl'].append(reward[0]+reward[1])
        self.trajectory['ipl'].append(reward[2])
        self.trajectory['hc'].append(reward[3])

        state = np.array([])
        state = np.append(state,orders)
        state = np.append(state,self.inventory)
        state = np.append(state,self.asset.price)
        state = np.append(state,spcurve)
        state = np.append(state,reward[2])
        self.trajectory['state'].append(state)

        self.iter += 1
        if self.iter >= self.max_iter:
            self.done = True
        return(state,np.array(reward).sum(),self.done,
            {'trade':sum(traded)})

    def reset(self):

        self.asset.reset()
        self.invstrs.reset()
        self.competitor.reset()
        self.reset_trajectory()
        self.done = False
        self.iter = 0 
        self.inventory = 0 

    def reset_trajectory(self):
        self.trajectory = {"s":[],"a":[],"i":[],"h":[],
            "t":[],"r":[],'state':[],'spl':[],'ipl':[],'hc':[]}


    def generate_initial_states(self):

        bid_s = np.arange(0,20) * 1e-5
        ask_s = np.arange(0,20) * 1e-5
        trades = np.zeros(len(self.invstrs.orders))
        inventory=0
        price = 1.
        inventorypl=0.
        
        st = np.array([])
        st = np.append(st,trades)
        st = np.append(st,inventory)
        st = np.append(st,price)
        st = np.append(st,bid_s)
        st = np.append(st,ask_s)
        st = np.append(st,inventorypl)

        return(st)

    def get_trajectory(self):

        trajectory = self.trajectory
        trajectory['s'] = np.array(trajectory['s'],dtype=np.float32)
        trajectory['a'] = np.array(trajectory['a'],dtype=np.float32)
        trajectory['i'] = np.array(trajectory['i'],dtype=np.float32)
        trajectory['h'] = np.array(trajectory['h'],dtype=np.float32)
        trajectory['t'] = np.array(trajectory['t'],dtype=np.float32)
        trajectory['r'] = np.array(trajectory['r'],dtype=np.float32)
        trajectory['state'] = np.array(trajectory['state'],dtype=np.float32)
        trajectory['spl'] = np.array(trajectory['spl'],dtype=np.float32)
        trajectory['ipl'] = np.array(trajectory['ipl'],dtype=np.float32)
        trajectory['hc'] = np.array(trajectory['hc'],dtype=np.float32)

        return trajectory


    def render(self,mode='human',close=False):

        print('this function is not developped')

    def seed(self):

        print('_seed is not defined yet')

# %%

if __name__ == '__main__' :

    s = stock()
    p=[];sp=[];v=[]
    for i in range(30):
        s.price_drift()
        s.set_lob()
        s.convert_to_spread_curve()
        p.append(s.price)
        sp.append(s.ask_price[0])
        v.append(s.ask_v[0])

    pd.Series(p).plot()
    pd.Series(sp).plot()


# %%
