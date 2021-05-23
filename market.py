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

        self.ask_price = self.price * np.ones(10) + \
            (s0/2.+np.arange(10)*0.0001)
        self.bid_price = self.price * np.ones(10) - \
            (s0/2.+np.arange(10)*0.0001)

        # Volume
        if seed is not None:
            v0 = gamma.rvs(1.,random_state=seed)
        else :
            v0 = gamma.rvs(1.)

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

class investors:

    def __init__(self,ninvestors=20):

        self.ninvestors=ninvestors
        self.orders = np.ones(self.ninvestors)
        self.sell = 0
        self.buy = 0 

    def reset(self):

        self.generate_orders()

    def generate_orders(self):

        self.orders = np.random.choice([-1.,1.],
            self.ninvestors
        )

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

    def __init__(self,competitor_type:str='random',
        ninvestor:int=20, max_iter = 10000
    ):

        self.max_iter = 10000
        self.iter = 0 
        self.done = False

        self.asset = stock()
        self.invstrs = investors(ninvestors=ninvestor)
        self.competitor = agenet_dict[competitor_type](
                [-1e-1,1e-1]
            )


    def step(self,action:tuple):

        '''

            Parameters
            __________
            action: tuple
                (bid_spread, ask_spread,inventory)
        '''

        # calc reward
        reward=[0,0,0]
        traded = [0,0]

        # inventory PL
        reward[2] = action[2] * (self.asset.price - self.asset.price_old)

        # Spread PL
        #bid
        if self.competitor.eps_bid > action[0] :
            reward[0] = self.asset.ref_spread_curve_bid[
                self.invstrs.sell
            ] * (1.+action[0])
            traded[0]=self.invstrs.sell # investor sell agent buy
        #ask
        if self.competitor.eps_ask > action[1] :
            reward[1] = self.asset.ref_spread_curve_ask[
                self.invstrs.buy
            ] * (1.+action[1])
            traded[1]=-self.invstrs.buy # investor buy agent sell

        observation = {
            'bid_s':self.asset.ref_spread_curve_bid,
            'bid_vcum':self.asset.bid_vcum,
            'ask_s':self.asset.ref_spread_curve_ask,
            'ask_vcum':self.asset.ask_vcum
        }

        # update stock
        self.asset.price_drift()
        self.asset.set_lob(lam=10.)
        self.asset.convert_to_spread_curve()

        # update investors
        self.invstrs.generate_orders()

        # udpate competitor
        self.competitor.set_offer_spread()

        self.iter += 1

        if self.iter >= self.max_iter:
            self.done = True

        return(observation,np.array(reward).sum(),self.done,
            {'trade':sum(traded)})

    def reset(self):

        self.asset.reset()
        self.invstrs.reset()
        self.competitor.reset()

    def render(self,mode='human',close=False):

        print('this function is not developped')

    def seed(self):

        print('_seed is not defined yet')

# %%
