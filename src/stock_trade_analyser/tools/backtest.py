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
        logger.info(f"Strategy columns: {list(strategy.columns)}")
        logger.info(f"Strategy shape: {strategy.shape}")
        # Vectorized: returns are the elementwise product of the bar-to-bar
        # close diff and the position held.
        #
        # `np.diff` yields n-1 values while `position` has n, and the original
        # loops paired them by position (diff[i] * position[i]) -- i.e. each
        # return is multiplied by the position at the *start* of the interval.
        # That alignment is preserved by truncating position to the diff length.
        returns = np.diff(strategy['close'].to_numpy(dtype='float64'))
        if len(returns) == 0:
            logger.warning(f"No strategy returns calculated for {ticker}")
            return 0

        positions = strategy['position'].to_numpy(dtype='float64')[:len(returns)]
        st_returns = returns * positions

        logger.info(f"st_strategy_ret length: {len(st_returns)}")
        investment_value = 100000
        number_of_stocks = floor(investment_value/strategy['close'].iloc[-1])
        st_investment_ret = number_of_stocks * st_returns

        total_investment_ret = round(float(st_investment_ret.sum()), 2)
        profit_percentage = floor((total_investment_ret/investment_value)*100)
        # print(f'Profit gained from th ̰e ST strategy by investing $100k in {ticker} : {total_investment_ret}')
        # print(f'Profit percentage of the ST strategy  for {ticker}: {profit_percentage}%')
        logger.info(f'Profit gained from the ST strategy by investing $100k in {ticker} : {total_investment_ret}')
        logger.info(f'Profit percentage of the ST strategy for {ticker}: {profit_percentage}%')
        logger.info(f"Finished Backtest for {ticker}")
        return profit_percentage
    
    