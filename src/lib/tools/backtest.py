import pandas as pd
import numpy as np
from math import floor
from termcolor import colored as cl
import logging

logger = logging.getLogger(__name__)

class BackTest:
    def __init__(self,):
        print()
        
    def back_test(self, ticker, strategy):
        logger.info(f"Starting Backtest for {ticker}")
        data_ret = pd.DataFrame(np.diff(strategy['close'])).rename(columns = {0:'returns'})
        st_strategy_ret = []

        for i in range(len(data_ret)):
            returns = data_ret['returns'][i]*strategy['position'][i]
            st_strategy_ret.append(returns)
            
        
        st_strategy_ret_df = pd.DataFrame(st_strategy_ret).rename(columns = {0:'st_returns'})
        investment_value = 100000
        number_of_stocks = floor(investment_value/strategy['close'][-1])
        st_investment_ret = []

        for i in range(len(st_strategy_ret_df['st_returns'])):
            returns = number_of_stocks*st_strategy_ret_df['st_returns'][i]
            st_investment_ret.append(returns)

        st_investment_ret_df = pd.DataFrame(st_investment_ret).rename(columns = {0:'investment_returns'})
        total_investment_ret = st_investment_ret_df['investment_returns'].sum().round(2)
        profit_percentage = floor((total_investment_ret/investment_value)*100)
        # print(f'Profit gained from th ̰e ST strategy by investing $100k in {ticker} : {total_investment_ret}')
        # print(f'Profit percentage of the ST strategy  for {ticker}: {profit_percentage}%')
        logger.info(f'Profit gained from the ST strategy by investing $100k in {ticker} : {total_investment_ret}')
        logger.info(f'Profit percentage of the ST strategy for {ticker}: {profit_percentage}%')
        logger.info(f"Finished Backtest for {ticker}")
        return profit_percentage
    
    