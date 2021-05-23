
#%%

import numpy as np
import pandas as pd
from scipy.stats import gamma

# %%

class MMAgent:

    def __init__(self):

        self.inventory = 0 #inventory
        self.inventoryPL = 0.
        self.SpredPL = 0.
        self.TotalPL = 0
        self.tradelog = []
        self.eps_bid = 0.
        self.eps_ask = 0.
        

    def set_offer_spread(self):
        pass


    def calc_SpredPL(self,newtrade:tuple,
        spred:tuple
    ):

        out = 0.
        for v,s in enumerate(newtrade,spred):
            out += v*s

        return(out)

    def calc_inventoryPL(self,price_diff:float):

        return(self.inventory*price_diff)

    def calc_TotalPL(self,newtrade,spread,price_diff):

        self.SpredPL = self.calc_SpredPL(newtrade,spread)
        self.inventoryPL = self.calc_inventoryPL(price_diff)
        self.TotalPL = self.SpredPL+self.inventoryPL

        return(self.TotalPL)



# %%

class RandomAgent(MMAgent):

    def __init__(self,eps:tuple):

        """

            Parameters
            __________

            eps: tuple
                spread epsilon, (min,max)

            Return
            ______
            None
        """

        super(RandomAgent,self).__init__()

        self.emin=eps[0]
        self.emax=eps[1]

    def reset(self):

        self.set_offer_spread()

    def set_offer_spread(self):

        self.eps_bid = np.random.uniform(
            low=self.emin,high=self.emax
            )
        self.eps_ask = np.random.uniform(
            low=self.emin,high=self.emax
            )



    
