# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:42:58 2020

@author: s2230494
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd


class MyEnv():
    TRAINING_MODE = -1
    VALIDATION_MODE = 0
    TEST_MODE = 1

    def __init__(self, verbose=False):
        # initialize environment
        super().__init__()

        self.init = True
        self.verbose = verbose
        self.trade_fees = True
        
#        self.state_min = np.array([0,0,0,0,0])
#        self.state_max = np.array([2,2,2,1000,10000])
        
        # Eine bestimmte Aktie initialisieren
#        kennziffer = kennziffer
#        stock = yf.Ticker(kennziffer)
#        self.df = stock.history(period="max")
        
        # Mehrere Aktien initialisieren
        #stock_list = ['^GDAXI' ,'DAI.DE', 'FME.DE', 'BAYN.DE', 'ALV.DE', 'TKA.DE', 'LIN.DE', 'EOAN.DE', 'DPW.DE', 'MRK.DE', 'FRE.DE', 'DBK.DE', '1COV.DE', 'BAS.DE', 'BMW.DE', 'VNA.DE', 'VOW3.DE', 'DB1.DE', 'RWE.DE', 'MUV2.DE', 'BEI.DE', 'SAP.DE', 'LHA.DE', 'CON.DE', 'SIE.DE', 'HEN3.DE', 'HEI.DE', 'ADS.DE', 'IFX.DE', 'DTE.DE', 'WDI.DE'] # DAX 09.08.20   
#        stock_list = ['DAI.DE', 'FME.DE', 'BAYN.DE', 'ALV.DE', 'EOAN.DE', 'VOW3.DE','RWE.DE','LHA.DE','CON.DE', 'WDI.DE'] # set funktioniert    
#        stock_list = ['DAI.DE', 'BAYN.DE', 'ALV.DE',  'EOAN.DE', 'DPW.DE', 'FRE.DE', 'VOW3.DE', 'DB1.DE', 'RWE.DE', 'MUV2.DE', 'SAP.DE', 'LHA.DE', 'CON.DE'] #  aus dem DAX  
        stock_list = ['DAI.DE', 'BAYN.DE', '1COV.DE', 'BAS.DE', 'BMW.DE', 'VNA.DE', 'VOW3.DE', 'DB1.DE', 'RWE.DE', 'MUV2.DE', 'SAP.DE', 'LHA.DE', 'CON.DE', 'SIE.DE', 'HEN3.DE', 'HEI.DE', 'ADS.DE', 'IFX.DE', 'DTE.DE', 'WDI.DE'] # try
        
        # DJI
#        DJI_stock_list = ['PFE', 'CSCO', 'WMT', 'JNJ', 'INTC', 'MSFT', 'MRK', 'WBA', 'PG', 'VZ', 'V', 'MMM', 'AAPL', 'HD', 'UNH', 'IBM', 'GS', 'CVX', 'DIS', 'RTX', 'TRV', 'JPM', 'XOM', 'AXP', 'BA']

        # Eurostoxx - ohne deutsche Aktien
#        Euro_stock_list = ['ITX.MC', 'CA.PA', 'BBVA.MC', 'OR.PA', 'BMW.DE', 'MC.PA', 'ABI.BR', 'IBE.MC', 'G.MI', 'SAF.PA', 'INGA.AS', 'PHIA.AS', 'ENI.MI', 'AI.PA', 'SAN.PA', 'SU.PA', 'ORA.PA', 'ASML.AS', 'ENEL.MI', 'AIR.PA'] # unvollständig 27.08.20

#        stock_list += DJI_stock_list # Dax Auswahl + DJI
#        stock_list += Euro_stock_list # Dax Auswahl + Euro
        
        stock_list = ['DAI.DE','DAI.DE'] # nur eine Aktie
        
        self.stock_list = stock_list # for debugging
        
        
        for idx, kennziffer in enumerate(stock_list):
            stock = yf.Ticker(kennziffer)
            df = stock.history(period="max")
            if idx==0:
                self.df_merge = df.reset_index()['Close'].rename(kennziffer)
                print(self.df_merge.shape)
            else:
#                self.df_merge = pd.concat([self.df_merge, df.reset_index()['Close'].rename(kennziffer)], axis=1, join='inner')
                self.df_merge = pd.concat([self.df_merge, df.reset_index()['Close'].rename(kennziffer)], axis=1, join='outer')
                print(self.df_merge.shape)
        
        # Rolling mean
        self.rolling_mean_200 = self.df_merge.rolling(window=200).mean()
        self.rolling_mean_20 = self.df_merge.rolling(window=20).mean()
        self.rolling_mean_5 = self.df_merge.rolling(window=5).mean()
               
        # RSI
#        self.RSI5 = self.calc_RSI(window_length=5)
#        self.RSI20 = self.calc_RSI(window_length=20)
#        self.RSI250 = self.calc_RSI(window_length=250)
        self.RSI_250 = self.calc_RSI(self.df_merge, window_length = 250)
        self.RSI_20 = self.calc_RSI(self.df_merge, window_length = 20)
        self.RSI_5 = self.calc_RSI(self.df_merge, window_length = 5)
        
        # MACD
        exp1 = self.df_merge.ewm(span=12, adjust=False).mean()
        exp2 = self.df_merge.ewm(span=26, adjust=False).mean()

        self.macd = exp1-exp2
        self.exp3 = self.macd.ewm(span=9, adjust=False).mean()
        
        self.macd_k0 = (self.macd<0).astype(int) # binäres Feature
        self.macd_k_exp3 = (self.macd<self.exp3).astype(int) # binäres Feature       
        
        # Letztes alltime high
        window_alltime = 30
#        self.alltime_high_bit = self.df_merge.rolling(window_alltime).max()==self.df_merge.cummmax()
        self.alltime_high_bit = self.df_merge.rolling(window_alltime).max().values==np.maximum.accumulate(self.df_merge.values) # müsste eigentlich mit pandas cummax gehen        

        
        # Zustand initialisieren
#        self.init_day = init_day
#        self.end_day = len(self.df['Close'].values)-2
        self._last_observation = self.reset()


    def step(self, action):
        

        # 1. state auslesen
#        agent_state = self._last_observation*(self.state_max - self.state_min) + self.state_min
#        aktien = agent_state[-2]
        aktien = self.aktien
        cash = self.cash
#        cash = agent_state[-1]
        if self.verbose:
            print(cash, '€ cash')
        
        # 2. Aktienpreis auslesen
#        stock_price = self.df['Close'].values[self.day] 
#        stock_price = self.df_merge.iloc[self.day,self.stock_id]
#        stock_price = self.df_merge.iloc[self.day,self.stock_id]
        stock_price = self.df_merge.iloc[self.valid_day[self.day],self.stock_id]
        
        self.stock_price = stock_price
        
        # Summe heute
#        value = depot_+cash_
        value = aktien*stock_price + cash # richtig?
               
        # 3. action ausführen
        if action>0:
#            invest_summe = np.max([cash, 0])*action[0] # kauf
            invest_summe = np.max([cash, 0])*self.action_activation(action[0]) # kauf
            fees = 0
#            fees = self.trade_fees*(np.max([np.abs(invest_summe)*0.0025, np.sign(np.abs(invest_summe))*5]) + np.sign(np.abs(invest_summe))*4.95) #  Gebühren
#            if cash<fees: # passt das?
#                invest_summe -= fees # Um auch Handelsgebühren zu decken
            if self.trade_fees:
                min_kaufpreis = stock_price + (np.max([np.abs(stock_price)*0.0025, np.sign(np.abs(stock_price))*5]) + np.sign(np.abs(stock_price))*4.95)
                if invest_summe<=min_kaufpreis: # nicht genügend Cash für eine Aktie
                    invest_summe = 0
                    fees = 0
                else:
                    fees = np.max([(invest_summe-4.95)/1.0025-invest_summe, 9.95])
#                    fees = invest_summe*0.01 # try
                    invest_summe -= fees 
        else:
#            invest_summe = aktien*stock_price*action[0] # verkauf
            invest_summe = aktien*stock_price*self.action_activation(action[0]) # verkauf
            fees = self.trade_fees*(np.max([np.abs(invest_summe)*0.0025, np.sign(np.abs(invest_summe))*5]) + np.sign(np.abs(invest_summe))*4.95) #  Gebühren
#            fees = self.trade_fees*np.abs(invest_summe)*0.01
        if self.verbose:
            print(invest_summe, '€ sollen investiert werden')
            print(fees, '€ Gebühren')
            
        # Gebühren
        # Mit Summe - Gebühren können tatsächlich Aktien gekauft werden
        
        # Untere Schranke -> kann eine Aktie gekauft werden?
            
        # 4. Kauf ausführen
        aktien_kauf = np.floor(invest_summe/stock_price)
        if self.verbose:
            print(aktien_kauf, 'Aktien können erworben werden')
            print('')
        
        aktien_ = aktien_kauf + aktien # Anzahl Aktien
        self.aktien = aktien_ # Aktien Update
        if self.verbose:
            print(aktien_, 'Aktien liegen nun im Depot')
        
#        depot_ = aktien_*stock_price # Neue Depot Summe
        
#        fees = self.trade_fees*(np.max([np.abs(invest_summe)*0.0025, np.sign(np.abs(invest_summe))*5]) + np.sign(np.abs(invest_summe))*4.95) #  Gebühren
        if self.verbose:
            print('Gebühren: ', fees, ' €')
        
        cash_ = cash - np.floor(invest_summe/stock_price)*stock_price - fees
        self.cash = cash_ # update
        if self.verbose:
            print('neuer Cashbestand: ', self.cash)
                
        # Summe morgen
#        stock_price_ = self.df['Close'].values[self.day+1]
#        stock_price_ = self.df_merge.iloc[self.day+1,self.stock_id]
        stock_price_ = self.df_merge.iloc[self.valid_day[self.day+1],self.stock_id]
        value_ = aktien_*stock_price_ + cash_
        self.value_ = value_ # um Wert von außen abzugreifen
        
        # Reward berechnen
#        reward = value_ - value
#        reward = (value_ - value)/stock_price_ # normiert -> unlogisch?
        reward = (value_ - value)/value_*100 # normiert - Änderung in %
        
        if self.verbose:
            print('reward:', value_ - value, (value_ - value)/value_)
        

        
        # 4. Geupdateten State zusammensetzen
        self.day += 1 # einen Tag inkrementieren
        
        if np.isnan(value_):
            print(self.stock_id, self.day, self.start_day,stock_price, stock_price_, cash, cash_, value, value_, aktien, aktien_, invest_summe, fees, action, self.investment_rate)
#            reward = 0
#            broke = True
#            print('data error')
        # new
        if value_>=0:
            broke = False
            self.investment_rate = aktien_*stock_price_/value_
        else:
            print('broke')
            broke = True
            self.investment_rate = 1
        
        
        if self.cash<self.df_merge.iloc[self.valid_day[self.day],self.stock_id]:
             self.stock_cash_ratio = 1.
        else:
             self.stock_cash_ratio = self.df_merge.iloc[self.valid_day[self.day],self.stock_id]/self.cash # prüfen

        
        state = self.feature_scaling(self.state_fct(day=self.day, investment_rate=self.investment_rate, stock_cash_ratio=self.stock_cash_ratio))
#        state = self.feature_scaling(self.state_fct(day=self.day, investment_rate=self.investment_rate))#.astype(float)
        
        
        self._last_observation = state
        
        # 5. Überprüfen ob Ende erreicht
#        if self.day>(len(self.df['Close'].values)-2):
        if self.day>self.end_day or broke:
            done = True
        else:
            done = False  

        return self._last_observation, reward, done

    
    def reset(self, training=False, stock_id=0):
#        self.aktien = 0.
#        self.cash = 10000.
        
#        stock_id = 0
              
        if training:
#            self.day = np.random.randint(low=300, high=len(self.df['Close'].values)-300)
#            self.end_day = np.random.randint(low=self.day+100, high=len(self.df['Close'].values)-100)
            
            # choose random stock
            self.stock_id = np.random.randint(low=1, high=self.df_merge.shape[1]) # um 0 mit Test auszuklammern
#            self.stock_id = stock_id
            
            # look for valid data
            self.valid_day = np.where(self.df_merge.iloc[:,self.stock_id].notna())[0] # an welchen Tagen sind Daten eingetragen
            
            # Einstiegstag bestimmen
#            self.day = np.random.randint(low=self.valid_day[0]+300, high=self.valid_day[-1]-300)
#            self.day = np.random.randint(low=self.valid_day[300], high=self.valid_day[-300])
            self.day = np.random.randint(low=250, high=len(self.valid_day)-300)
            self.start_day = self.day # für debugging
            
#            self.end_day =  np.random.randint(low=self.day+100, high=self.valid_day[-1]-100)   
#            self.end_day =  np.random.randint(low=self.day, high=self.valid_day[-100]) 
            self.end_day =  np.random.randint(low=self.day, high=len(self.valid_day)-100)
            
            if self.verbose: # für debugging
                print(self.stock_id, self.day, self.end_day) 
            
#            self.stock_cash_ratio = self.df_merge.iloc[self.day,self.stock_id]/self.cash # prüfen
            # Cash und Aktienanzahl bestimmen
            stock_price = self.df_merge.iloc[self.valid_day[self.day],self.stock_id]
            
            self.cash_init = np.random.randint(low=100, high = 100000) # mit viel viel cash anfangen
            
            if self.cash_init/stock_price>1:
                self.aktien = np.random.randint(low=0, high=np.floor(self.cash_init/stock_price))
            else:
                self.aktien = 0
            self.cash = self.cash_init - self.aktien*stock_price
            
            self.investment_rate = self.aktien*stock_price/(self.aktien*stock_price+self.cash)
           
            
        else:
            self.aktien = 0.
            self.cash = 10000.
#            self.day = self.init_day
#            self.end_day = len(self.df['Close'].values)-2
            self.stock_id = stock_id
            self.valid_day = np.where(self.df_merge.iloc[:,self.stock_id].notna())[0]
#            self.day = self.valid_day[0]+300
#            self.day = self.valid_day[300]
            self.day = 200
#            self.end_day =  self.valid_day[-1]-2  
#            self.end_day =  self.valid_day[-3]
            self.end_day = len(self.valid_day)-2
            self.investment_rate = 0
            
#            self.stock_cash_ratio = self.df_merge.iloc[self.day,self.stock_id]/self.cash # prüfen
        
#        self.value_ = 0 # ???
#        self.stock_cash_ratio = self.df['Close'].values[self.day]/self.cash
        if self.cash>0:
            self.stock_cash_ratio = self.df_merge.iloc[self.valid_day[self.day],self.stock_id]/self.cash #
        else:
            self.stock_cash_ratio = 1
            
        if self.verbose:
            print(self.aktien, self.cash)
            
        
        
        return self.feature_scaling(self.state_fct(day=self.day, investment_rate=0, stock_cash_ratio=self.stock_cash_ratio))
#        return self.feature_scaling(self.state_fct(day=self.day, investment_rate=self.investment_rate))

    
    def state_fct(self, day=300, investment_rate=0, stock_cash_ratio=0):
        # Moving average
#        ma5_n = np.mean(self.df['Close'].values[-6+day:-1+day])/self.df['Close'].values[day]#-0.5
#        ma20_n = np.mean(self.df['Close'].values[-21+day:-1+day])/self.df['Close'].values[day]#-0.5
#        ma200_n = np.mean(self.df['Close'].values[-201+day:-1+day])/self.df['Close'].values[day]#-0.5
        
        # Momentum
#        momentum20 = self.df['Close'].values[day]/self.df['Close'].values[day-20]
#        momentum50 = self.df['Close'].values[day]/self.df['Close'].values[day-50]
#        momentum250 = self.df['Close'].values[day]/self.df['Close'].values[day-250]  
        
        # RSI
#        RSI = self.RSI[day-1] # überprüfen
#        RSI5 = self.RSI5[day-1] # überprüfen
#        RSI20 = self.RSI20[day-1] # überprüfen
#        RSI250 = self.RSI250[day-1] # überprüfen
        
        # für multiple
        # MA
#        ma200_n = self.rolling_mean_200.iloc[self.valid_day[day],self.stock_id]/self.df_merge.iloc[self.valid_day[day],self.stock_id] - 0.5
        ma200_n = self.df_merge.iloc[self.valid_day[day],self.stock_id]/self.rolling_mean_200.iloc[self.valid_day[day],self.stock_id] - 0.5
#        ma20_n = self.rolling_mean_20.iloc[self.valid_day[day],self.stock_id]/self.df_merge.iloc[self.valid_day[day],self.stock_id] - 0.5
        ma20_n = self.df_merge.iloc[self.valid_day[day],self.stock_id]/self.rolling_mean_20.iloc[self.valid_day[day],self.stock_id] - 0.5
#        ma5_n = self.rolling_mean_5.iloc[self.valid_day[day],self.stock_id]/self.df_merge.iloc[self.valid_day[day],self.stock_id] - 0.5
        ma5_n = self.df_merge.iloc[self.valid_day[day],self.stock_id]/self.rolling_mean_5.iloc[self.valid_day[day],self.stock_id] - 0.5
        
        #RSI
        RSI250 = self.RSI_250[self.valid_day[day],self.stock_id]
        RSI20 = self.RSI_20[self.valid_day[day],self.stock_id]
        RSI5 = self.RSI_5[self.valid_day[day],self.stock_id]
#        RSI14 = self.RSI_14[self.valid_day[day],self.stock_id]

        
        # MACD
        macd_k0 = self.macd_k0.iloc[self.valid_day[day],self.stock_id]
        macd_k_exp3 = self.macd_k_exp3.iloc[self.valid_day[day],self.stock_id]
        
        # Alltime high
        high = self.alltime_high_bit[self.valid_day[day],self.stock_id]

        
        if np.any(np.isnan([ma200_n, ma20_n, ma5_n, RSI5, RSI20, RSI250, macd_k0, macd_k_exp3, high, investment_rate])):
            print(self.stock_id, day)
        
#        return np.array([ma200_n, RSI20, macd_k0, macd_k_exp3, investment_rate])
        return np.array([ma200_n, ma20_n, ma5_n, RSI5, RSI20, RSI250, macd_k0, macd_k_exp3, high, investment_rate])
    
    
 
    def feature_scaling(self, state):
        """
        Min-Max-Scaler: scale X' = (X-Xmin) / (Xmax-Xmin)
        :param state:
        :return: scaled state
        """
#        return (state - self.state_min) / (self.state_max - self.state_min)
        return state    
    
    def calc_RSI(self, close, window_length = 14):        
#        close = self.df['Close']
        delta = close.diff()
#        delta = delta[1:] 

        # Make the positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # Calculate the EWMA
        roll_up1 = up.ewm(span=window_length).mean()
        roll_down1 = down.abs().ewm(span=window_length).mean()

        # Calculate the RSI based on EWMA
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))
        
        return RSI1.values/100
    
    def action_activation(self, action):
        # doppel sigmoid 
        if action>0:
            return 1/(1 + np.exp(-action*20 + 10)) 
        else:
            return -(1/(1 + np.exp(action*20 + 10)))
        
    def seed(self, seed=0):
        pass
    
if __name__ == '__main__':
    
    env = MyEnv(verbose=True)
#    env.trade_fees = True
    # Visualisieren
#    plt.figure()
#    plt.plot(env.df.index, env.df['Close'])
#    plt.grid()
    
#    print(len(env.df.index))
       
    # state abfragen
    print(env._last_observation)    
#    print(env._last_observation[-1], '€ cash')
#    print(env._last_observation[-2], 'Aktien')
#    print(env.value_)
    
    # step durchführen
    print(env.step(action=np.array([0.])))
#    print(env._last_observation[-1], '€ cash')
#    print(env._last_observation[-2], 'Aktien')
    print(env.value_)

    # step durchführen
    print(env.step(action=np.array([0.1])))
#    print(env._last_observation[-1], '€ cash')
#    print(env._last_observation[-2], 'Aktien')
    print(env.value_)
    
    print(env.step(action=np.array([-1.])))
    print(env.value_)
    
    s = env.reset(training=True) 
    
    print(env.step(action=np.array([0.])))