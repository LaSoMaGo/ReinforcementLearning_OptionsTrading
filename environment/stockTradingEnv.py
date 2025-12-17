<<<<<<< HEAD
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import sys
import matplotlib.pyplot as plt
from IPython.display import display


INIT_MONEY = 1000_000
COMMISSION = 5
COMMISSION_RATE = 0.0003
TAX = 0.001 # Applied when selling stocks 



class stockTrading(gym.Env):
    metadata = {"render_modes":"human"}

    def __init__(self, 
                df,
                stock_price_df, 
                lookback_period ,
                device,
                starting_balance = INIT_MONEY):
        super(stockTrading, self).__init__()

        self.device = device 
        self.df = df
        self.stock_price_df = stock_price_df.values
        self.lookback_period = lookback_period
        self.init_money = starting_balance
        self.n_features = len(self.df.columns) * self.lookback_period

        # To retrieve basic size of action, observation_space and reward 
        self.action_space = spaces.Discrete(3) # (0) hold, (1) buy, (2)sell 
        self.observation_space = spaces.Box(low=-1, 
                                           high=1,
                                           shape= (self.n_features, ),
                                           dtype = np.float32)
        self.reward_range = (-np.inf, np.inf)

        # Pointer 
        self.t = 0 

        # Important finance info 
        self.min_commission = COMMISSION
        self.commission_rate = COMMISSION_RATE
        self.tax = TAX 


    def reset(self, seed = None, options=None):

        self.hold_money = self.init_money # reset holding money to initial money
        
        self.buy_price = 0                # previous stock buying price
        self.sell_price = 0               # previous stock selling price
        
        self.stock_value = 0              # total market value of stock
        self.total_value = 0              # total stock value + cash
        self.last_value = self.init_money # total value in t-1
        self.hold_num = 0                 # num of stocks currently hold
        self.hold_period = 0              # num of consecutive action 0 has been called
        self.total_profit = 0             # total profits so far
        self.reward = 0.0                   # reward at current timestamp
        
        self.consecutive_action = 0       # number of consecutive actions called 
        self.pre_action = -1              # action chosen in previous step
        
        # record the timing for each executed actions
        self.time_buy_stock = [] 
        self.time_sell_stock = [] 
        self.time_hold_stock = []
        
        # performance recorder
        self.profit_rate_account = [] 
        self.profit_rate_stock = []
        self.daily_return_account = []
        
        self.t = 0 
        
        # observation, info 
        return self.next_observation(self.t), {}
    
    def next_observation(self, t):
        day = t - self.lookback_period + 1

        historical = []

        # if the current timestamp < lookback period, pad with the first day data
        if t < self.lookback_period - 1:
            historical = np.concatenate((-(t - self.lookback_period + 1) * [self.df.values[0, :]], self.df.values[0 : t + 1, :]))
        else:
            historical = self.df.values[t - self.lookback_period + 1 : t + 1]
        
        # data at t+1
        if t < self.lookback_period - 2:
            historical2 = np.concatenate((-(t - self.lookback_period + 2) * [self.df.values[0, :]], self.df.values[0 : t + 2, :]))
        else:
            historical2 = self.df.values[t - self.lookback_period + 2 : t + 2]

        historical = (historical2 / (historical + 0.0001) - 1) 
        historical = historical.reshape(1, -1)
    
        return torch.FloatTensor(historical).to(self.device)# return states @ time t
    
    def buy_stock(self):

        # buy stocks in batches of 10 -> if the remaining money cannot buy 10 stocks at once,
        # then the buy action cannot be carried out
        buy_num = (self.hold_money // self.stock_price_df[self.t] // 10) * 10
        self.buy_price = self.stock_price_df[self.t]
        
        volume = self.stock_price_df[self.t] * buy_num # current stock price * trading Q

        commission = max(volume * self.commission_rate, self.min_commission)

        # if the commission cannot be paid, then reduce buying num by 10
        while commission + volume > self.hold_money and buy_num > 0:
            buy_num -= 10
            volume -= 10 * self.stock_price_df[self.t]
            commission = max(volume * self.commission_rate, self.min_commission)
        
        # if a buy action is executed, update the portfolio
        if buy_num > 0:
            self.hold_num += buy_num
            self.stock_value += volume
            
            # update holding money by lessing stock costs and commission fees
            self.hold_money -=  volume + commission 

            self.time_buy_stock.append(self.t) # record the buy action

    
    def sell_stock(self, sell_num):

        volume = sell_num * self.stock_price_df[self.t]

        self.sell_price = self.stock_price_df[self.t]

        commission = max(volume * self.commission_rate, self.min_commission)
        sell_tax = self.tax * volume
        
        # update holding money by adding incomes from stocks and lessing applicable commission and tax
        self.hold_money = self.hold_money + volume - commission - sell_tax
        self.hold_num = 0
        self.stock_value = 0
        self.time_sell_stock.append(self.t)

    def step(self, action):

        # check if the agent repeatedly calls the same action
        if action == self.pre_action:
            self.consecutive_action += 1
        else:
            self.consecutive_action = 0
            self.pre_action = action
        
        # carries out the action
        if action == 1:
            self.buy_stock()
            self.hold_period = 0

        elif action == 2 and self.hold_num > 0:
            self.hold_period = 0
            self.sell_stock(self.hold_num)
        else:
            self.time_hold_stock.append(self.t)
            self.hold_period += 1
        
        # current market value of holding stocks
        self.stock_value = self.stock_price_df[self.t] * self.hold_num
        
        self.total_value = self.stock_value + self.hold_money   # total portfolio value 
        self.total_profit = self.total_value - self.init_money  # profits above initial funds
        
        # basic reward
        self.reward = (self.total_value / self.last_value - 1)
        self.daily_return_account.append(self.reward)
        
        # scaled reward calculation to encourage "buy-low-sell-high"
        if action == 1 and self.sell_price != 0 and self.hold_num == 0:
            if self.reward > 0:
                self.reward /= (self.sell_price / self.stock_price_df[self.t])
            else:
                self.reward *= self.sell_price / self.stock_price_df[self.t]
        elif action == 2 and self.hold_num > 0:
            if self.reward > 0:
                self.reward *= self.stock_price_df[self.t] / self.buy_price
            else:
                self.reward /= (self.stock_price_df[self.t] / self.buy_price)
        
        # applies penalties on repeated actions
        if self.consecutive_action > 0:
            if self.pre_action == 1 or self.pre_action == 2:
                if self.reward > 0:
                    self.reward *= 0.25 ** self.consecutive_action
                else:
                    self.reward = self.reward * min(sys.float_info.max / 2, 4 ** self.consecutive_action)
            self.reward -= self.hold_period ** 1.1
        
        self.last_value = self.total_value
        
        self.profit_rate_account.append(self.total_value / self.init_money - 1)
        self.profit_rate_stock.append(self.stock_price_df[self.t] / self.stock_price_df[0] - 1)

        self.t = self.t + 1
        s_ = self.next_observation(self.t)

        #values: obs, reward, terminated, truncated, info.
        return s_, float(self.reward), self.t == len(self.df) - 2, False, {}


    def get_info(self):
        return self.time_sell_stock, self.time_buy_stock, self.time_hold_stock, self.profit_rate_account, self.profit_rate_stock, self.daily_return_account  
    
    def calc_sharpe(self):
        # calculation of sharpe ratio
        daily_return = np.array(self.daily_return_account)
        return np.mean(daily_return) / np.std(daily_return) * (252 ** 0.5)
    
    def render(self, save_name1, save_name2, options_data = False):

        time_sell_stock, time_buy_stock, time_hold_stock, profit_rate_account, profit_rate_stock, daily_return = self.get_info()
        
        total_gains = self.total_profit   # total gains in dollars
        roi = profit_rate_account[-1]     # final ROI
        sharpe = self.calc_sharpe()       # sharpe ratio

        
        
        if not options_data :
            # plot buy, sell, (hold) actions by time, on stock's trend
            # uncomment for plotting hold action
            stock_price = self.stock_price_df
            fig = plt.figure(figsize = (15,5))
            plt.plot(stock_price, color='#0C1844', lw=0.8)

            plt.plot(stock_price, '^', label = 'comprar', markevery = time_buy_stock, color='#006989', markersize=3)
            plt.plot(stock_price, 'v', label = 'vender', markevery = time_sell_stock, color='#C80036', markersize=3)
            # plt.plot(stock_price, '>', label = 'hold action', markevery = time_hold_stock, color='b', markersize=8)

            plt.title('Testeo Sobre Datos de Stock, Ganancias Totales %.2f, ROI %.2f%%, sharpe ratio %.2f'%(total_gains, roi * 100, sharpe))
            plt.legend()
            # plt.savefig(save_name1)
            plt.xlabel("Número de días")
            plt.ylabel("Precio de Cierre")
            display(fig)
            plt.close()

            fig = plt.figure(figsize = (15,5))
            # plot the stock performance comparisons
            plt.plot(profit_rate_account, label='Agente', color = '#FFB1B1')
            plt.plot(profit_rate_stock, label='Opcion', color = '#9B86BD')
            plt.legend()
            # plt.savefig(save_name2)
            plt.xlabel("Número de días")
            plt.ylabel("Taza de Retorno")
            display(fig)
            plt.close()

        else:

            # plot buy, sell, (hold) actions by time, on stock's trend
            # uncomment for plotting hold action
            stock_price = self.stock_price_df
            fig = plt.figure(figsize = (15,5))
            plt.plot(stock_price, color='#CA8787', lw=0.8)

            plt.plot(stock_price, '^', label = 'Call', markevery = time_buy_stock, color='#8576FF', markersize=3)
            plt.plot(stock_price, 'v', label = 'Put', markevery = time_sell_stock, color='#7BC9FF', markersize=3)
            # plt.plot(stock_price, '>', label = 'hold action', markevery = time_hold_stock, color='b', markersize=8)
        
            plt.title('Datos de Opciones Validacion, Ganancias Totales %.2f, ROI %.2f%%, sharpe ratio %.2f'%(total_gains, roi * 100, sharpe))
            plt.legend()
            # plt.savefig(save_name1)
            plt.xlabel("Número de días")
            plt.ylabel("Precio de Cierre")
            display(fig)
            plt.close()
        

            fig = plt.figure(figsize = (15,5))
            # plot the stock performance comparisons
            plt.plot(profit_rate_account, label='Agente', color = '#FFAF61')
            plt.plot(profit_rate_stock, label='Opcion', color = '#FF70AB')
            plt.legend()
            # plt.savefig(save_name2)
            plt.xlabel("Número de días")
            plt.ylabel("Taza de Retorno")
            display(fig)
            plt.close()
=======
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import sys
import matplotlib.pyplot as plt
from IPython.display import display


INIT_MONEY = 1000_000
COMMISSION = 5
COMMISSION_RATE = 0.0003
TAX = 0.001 # Applied when selling stocks 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class stockTrading(gym.Env):
    metadata = {"render_modes":"human"}

    def __init__(self, 
                df,
                stock_price_df, 
                lookback_period ,
                starting_balance = INIT_MONEY):
        super(stockTrading, self).__init__()

        
        self.df = df
        self.stock_price_df = stock_price_df.values
        self.lookback_period = lookback_period
        self.init_money = starting_balance
        self.n_features = len(self.df.columns) * self.lookback_period

        # To retrieve basic size of action, observation_space and reward 
        self.action_space = spaces.Discrete(3) # (0) hold, (1) buy, (2)sell 
        self.observation_space = spaces.Box(low=-1, 
                                           high=1,
                                           shape= (self.n_features, ),
                                           dtype = np.float32)
        self.reward_range = (-np.inf, np.inf)

        # Pointer 
        self.t = 0 

        # Important finance info 
        self.min_commission = COMMISSION
        self.commission_rate = COMMISSION_RATE
        self.tax = TAX 


    def reset(self, seed = None, options=None):

        self.hold_money = self.init_money # reset holding money to initial money
        
        self.buy_price = 0                # previous stock buying price
        self.sell_price = 0               # previous stock selling price
        
        self.stock_value = 0              # total market value of stock
        self.total_value = 0              # total stock value + cash
        self.last_value = self.init_money # total value in t-1
        self.hold_num = 0                 # num of stocks currently hold
        self.hold_period = 0              # num of consecutive action 0 has been called
        self.total_profit = 0             # total profits so far
        self.reward = 0.0                   # reward at current timestamp
        
        self.consecutive_action = 0       # number of consecutive actions called 
        self.pre_action = -1              # action chosen in previous step
        
        # record the timing for each executed actions
        self.time_buy_stock = [] 
        self.time_sell_stock = [] 
        self.time_hold_stock = []
        
        # performance recorder
        self.profit_rate_account = [] 
        self.profit_rate_stock = []
        self.daily_return_account = []
        
        self.t = 0 
        
        # observation, info 
        return self.next_observation(self.t), {}
    
    def next_observation(self, t):
        day = t - self.lookback_period + 1

        historical = []

        # if the current timestamp < lookback period, pad with the first day data
        if t < self.lookback_period - 1:
            historical = np.concatenate((-(t - self.lookback_period + 1) * [self.df.values[0, :]], self.df.values[0 : t + 1, :]))
        else:
            historical = self.df.values[t - self.lookback_period + 1 : t + 1]
        
        # data at t+1
        if t < self.lookback_period - 2:
            historical2 = np.concatenate((-(t - self.lookback_period + 2) * [self.df.values[0, :]], self.df.values[0 : t + 2, :]))
        else:
            historical2 = self.df.values[t - self.lookback_period + 2 : t + 2]

        historical = (historical2 / (historical + 0.0001) - 1) 
        historical = historical.reshape(1, -1)
    
        return torch.FloatTensor(historical).to(device)# return states @ time t
    
    def buy_stock(self):

        # buy stocks in batches of 10 -> if the remaining money cannot buy 10 stocks at once,
        # then the buy action cannot be carried out
        buy_num = (self.hold_money // self.stock_price_df[self.t] // 10) * 10
        self.buy_price = self.stock_price_df[self.t]
        
        volume = self.stock_price_df[self.t] * buy_num # current stock price * trading Q

        commission = max(volume * self.commission_rate, self.min_commission)

        # if the commission cannot be paid, then reduce buying num by 10
        while commission + volume > self.hold_money and buy_num > 0:
            buy_num -= 10
            volume -= 10 * self.stock_price_df[self.t]
            commission = max(volume * self.commission_rate, self.min_commission)
        
        # if a buy action is executed, update the portfolio
        if buy_num > 0:
            self.hold_num += buy_num
            self.stock_value += volume
            
            # update holding money by lessing stock costs and commission fees
            self.hold_money -=  volume + commission 

            self.time_buy_stock.append(self.t) # record the buy action

    
    def sell_stock(self, sell_num):

        volume = sell_num * self.stock_price_df[self.t]

        self.sell_price = self.stock_price_df[self.t]

        commission = max(volume * self.commission_rate, self.min_commission)
        sell_tax = self.tax * volume
        
        # update holding money by adding incomes from stocks and lessing applicable commission and tax
        self.hold_money = self.hold_money + volume - commission - sell_tax
        self.hold_num = 0
        self.stock_value = 0
        self.time_sell_stock.append(self.t)

    def step(self, action):

        # check if the agent repeatedly calls the same action
        if action == self.pre_action:
            self.consecutive_action += 1
        else:
            self.consecutive_action = 0
            self.pre_action = action
        
        # carries out the action
        if action == 1:
            self.buy_stock()
            self.hold_period = 0

        elif action == 2 and self.hold_num > 0:
            self.hold_period = 0
            self.sell_stock(self.hold_num)
        else:
            self.time_hold_stock.append(self.t)
            self.hold_period += 1
        
        # current market value of holding stocks
        self.stock_value = self.stock_price_df[self.t] * self.hold_num
        
        self.total_value = self.stock_value + self.hold_money   # total portfolio value 
        self.total_profit = self.total_value - self.init_money  # profits above initial funds
        
        # basic reward
        self.reward = (self.total_value / self.last_value - 1)
        self.daily_return_account.append(self.reward)
        
        # scaled reward calculation to encourage "buy-low-sell-high"
        if action == 1 and self.sell_price != 0 and self.hold_num == 0:
            if self.reward > 0:
                self.reward /= (self.sell_price / self.stock_price_df[self.t])
            else:
                self.reward *= self.sell_price / self.stock_price_df[self.t]
        elif action == 2 and self.hold_num > 0:
            if self.reward > 0:
                self.reward *= self.stock_price_df[self.t] / self.buy_price
            else:
                self.reward /= (self.stock_price_df[self.t] / self.buy_price)
        
        # applies penalties on repeated actions
        if self.consecutive_action > 0:
            if self.pre_action == 1 or self.pre_action == 2:
                if self.reward > 0:
                    self.reward *= 0.25 ** self.consecutive_action
                else:
                    self.reward = self.reward * min(sys.float_info.max / 2, 4 ** self.consecutive_action)
            self.reward -= self.hold_period ** 1.1
        
        self.last_value = self.total_value
        
        self.profit_rate_account.append(self.total_value / self.init_money - 1)
        self.profit_rate_stock.append(self.stock_price_df[self.t] / self.stock_price_df[0] - 1)

        self.t = self.t + 1
        s_ = self.next_observation(self.t)

        #values: obs, reward, terminated, truncated, info.
        return s_, float(self.reward), self.t == len(self.df) - 2, False, {}


    def get_info(self):
        return self.time_sell_stock, self.time_buy_stock, self.time_hold_stock, self.profit_rate_account, self.profit_rate_stock, self.daily_return_account  
    
    
    
    def get_info(self):
        return self.time_sell_stock, self.time_buy_stock, self.time_hold_stock, self.profit_rate_account, self.profit_rate_stock, self.daily_return_account  
    
    def calc_sharpe(self):
        # calculation of sharpe ratio
        daily_return = np.array(self.daily_return_account)
        return np.mean(daily_return) / np.std(daily_return) * (252 ** 0.5)
    
    def render(self, save_name1, save_name2):

        time_sell_stock, time_buy_stock, time_hold_stock, profit_rate_account, profit_rate_stock, daily_return = self.get_info()
        
        total_gains = self.total_profit   # total gains in dollars
        roi = profit_rate_account[-1]     # final ROI
        sharpe = self.calc_sharpe()       # sharpe ratio

        stock_price = self.stock_price_df
        fig = plt.figure(figsize = (15,5))
        plt.plot(stock_price, color='#CA8787', lw=0.8)
        
        # plot buy, sell, (hold) actions by time, on stock's trend
        # uncomment for plotting hold action
        plt.plot(stock_price, '^', label = 'Call', markevery = time_buy_stock, color='#8576FF', markersize=3)
        plt.plot(stock_price, 'v', label = 'Put', markevery = time_sell_stock, color='#7BC9FF', markersize=3)
        # plt.plot(stock_price, '>', label = 'hold action', markevery = time_hold_stock, color='b', markersize=8)
        
        plt.title('Ganancias Totales %.2f, ROI %.2f%%, sharpe ratio %.2f'%(total_gains, roi * 100, sharpe))
        plt.legend()
        plt.savefig(save_name1)
        plt.xlabel("Número de días")
        plt.ylabel("Precio de Cierre")
        display(fig)
        plt.close()
        

        fig = plt.figure(figsize = (15,5))
        # plot the stock performance comparisons
        plt.plot(profit_rate_account, label='Agente', color = '#FFAF61')
        plt.plot(profit_rate_stock, label='Opcion', color = '#FF70AB')
        plt.legend()
        plt.savefig(save_name2)
        plt.xlabel("Número de días")
        plt.ylabel("Taza de Retorno")
        display(fig)
        plt.close()
>>>>>>> 57256aacc2facb2ef40888678f92e3e94d08e1af
        