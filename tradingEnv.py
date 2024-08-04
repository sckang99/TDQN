# coding=utf-8

"""
Goal: Implement a trading environment compatible with OpenAI Gym.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import os
import gym
import math
import numpy as np
import ta
import pandas as pd

from matplotlib import pyplot as plt

from dataDownloader import AlphaVantage
from dataDownloader import YahooFinance
from dataDownloader import CSVHandler
from fictiveStockGenerator import StockGenerator

# to disable warning sign
pd.options.mode.copy_on_write = True

###############################################################################
################################ Global variables #############################
###############################################################################

# Boolean handling the saving of the stock market data downloaded
saving = True

# Variable related to the fictive stocks supported
fictiveStocks = ('LINEARUP', 'LINEARDOWN', 'SINUSOIDAL', 'TRIANGLE')

# Tuple of Technical Analysis
technicalAnalysis = ('MACD', 'CCI', 'RSI', 'ADX')


###############################################################################
############################## Class TradingEnv ###############################
###############################################################################

class TradingEnv(gym.Env):
    """
    GOAL: Implement a custom trading environment compatible with OpenAI Gym.
    
    VARIABLES:  - data: Dataframe monitoring the trading activity.
                - state: RL state to be returned to the RL agent.
                - reward: RL reward to be returned to the RL agent.
                - done: RL episode termination signal.
                - t: Current trading time step.
                - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - stateLength: Number of trading time steps included in the state.
                - numberOfShares: Number of shares currently owned by the agent.
                - transactionCosts: Transaction costs associated with the trading
                                    activity (e.g. 0.01 is 1% of loss).
                - technical Analysis
                                
    METHODS:    - __init__: Object constructor initializing the trading environment.
                - reset: Perform a soft reset of the trading environment.
                - step: Transition to the next trading time step.
                - render: Illustrate graphically the trading environment.
    """

    def __init__(self, marketSymbol, startingDate, endingDate, testDate, money, stateLength=30,
                 transactionCosts=0, startingPoint=0):
        """
        GOAL: Object constructor initializing the trading environment by setting up
              the trading activity dataframe as well as other important variables.
        
        INPUTS: - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - money: Initial amount of money at the disposal of the agent.
                - stateLength: Number of trading time steps included in the RL state.
                - transactionCosts: Transaction costs associated with the trading
                                    activity (e.g. 0.01 is 1% of loss).
                - startingPoint: Optional starting point (iteration) of the trading activity.
                - technical Analysis; List of technical analysis 
        
        OUTPUTS: /
        """

        # CASE 1: Fictive stock generation
        if(marketSymbol in fictiveStocks):
            stockGeneration = StockGenerator()
            if(marketSymbol == 'LINEARUP'):
                self.data = stockGeneration.linearUp(startingDate, endingDate)
            elif(marketSymbol == 'LINEARDOWN'):
                self.data = stockGeneration.linearDown(startingDate, endingDate)
            elif(marketSymbol == 'SINUSOIDAL'):
                self.data = stockGeneration.sinusoidal(startingDate, endingDate)
            else:
                self.data = stockGeneration.triangle(startingDate, endingDate)
 
        # CASE 2: Real stock loading
        else:
            # Check if the stock market data is already present in the database
            csvConverter = CSVHandler()
            csvName1 = "".join(['Data/', marketSymbol, '_', startingDate, '_', endingDate])
            exist1 = os.path.isfile(csvName1 + '.csv')
            csvName2 = "".join(['Data/', marketSymbol])
            exist2 = os.path.isfile(csvName2 + '.csv')
            
            # If affirmative, load the stock market data from the database
            if(exist1 | exist2):
                if (exist1):
                    self.data = csvConverter.CSVToDataframe(csvName1)
                else:
                    data = csvConverter.CSVToDataframe(csvName2)
                    self.data = data.loc[(data.index >= startingDate) & (data.index <= endingDate), :]
            # Otherwise, download the stock market data from Yahoo Finance and save it in the database
            else:  
                downloader1 = YahooFinance()
                downloader2 = AlphaVantage()
                try:
                    self.data = downloader1.getDailyData(marketSymbol, startingDate, endingDate)
                except:
                    self.data = downloader2.getDailyData(marketSymbol, startingDate, endingDate)

                if saving == True:
                    csvConverter.dataframeToCSV(csvName1, self.data)

        # Interpolate in case of missing data
#        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
#        self.data.ffill(inplace=True)
#        self.data.bfill(inplace=True)
#        self.data.fillna(0, inplace=True)

        for taSymbol in technicalAnalysis:
            if(taSymbol == 'MACD'):
                self.data['MACD'] = ta.trend.macd(self.data['Close'], fillna=True)
            elif(taSymbol == 'CCI'):
                self.data['CCI'] = ta.trend.cci(self.data['High'], self.data['Low'], self.data['Close'], fillna=True)
            elif(taSymbol == 'RSI'):
                self.data['RSI'] = ta.momentum.rsi(self.data['Close'], fillna=True)
            elif(taSymbol == 'ADX'):
                 self.data['ADX'] = ta.trend.adx(self.data['High'], self.data['Low'], self.data['Close'], fillna=True)
    
        self.data.ffill(inplace=True)
        
        # Set the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:stateLength].tolist(),
                      self.data['Low'][0:stateLength].tolist(),
                      self.data['High'][0:stateLength].tolist(),
                      self.data['Volume'][0:stateLength].tolist()]
        for taSymbol in technicalAnalysis:
            if(taSymbol == 'MACD'):
                self.state.append(self.data['MACD'][0:stateLength].tolist())
            elif(taSymbol == 'CCI'):
                self.state.append(self.data['CCI'][0:stateLength].tolist())
            elif(taSymbol == 'RSI'):
                self.state.append(self.data['RSI'][0:stateLength].tolist())
            elif(taSymbol == 'ADX'):
                self.state.append(self.data['ADX'][0:stateLength].tolist())
        self.state.append([0])    
        self.reward = 0.
        self.done = 0

        # Set additional variables related to the trading activity
        self.marketSymbol = marketSymbol
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.testDate = testDate
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1

        # If required, set a custom starting point for the trading activity
        if startingPoint:
            self.setStartingPoint(startingPoint)


    def reset(self):
        """
        GOAL: Perform a soft reset of the trading environment. 
        
        INPUTS: /    
        
        OUTPUTS: - state: RL state returned to the trading strategy.
        """

        # Reset the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'].iloc[0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Reset the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:self.stateLength].tolist(),
                      self.data['Low'][0:self.stateLength].tolist(),
                      self.data['High'][0:self.stateLength].tolist(),
                      self.data['Volume'][0:self.stateLength].tolist()]
        for taSymbol in technicalAnalysis:
            if(taSymbol == 'MACD'):
                self.state.append(self.data['MACD'][0:self.stateLength].tolist())
            elif(taSymbol == 'CCI'):
                self.state.append(self.data['CCI'][0:self.stateLength].tolist())
            elif(taSymbol == 'RSI'):
                self.state.append(self.data['RSI'][0:self.stateLength].tolist())
            elif(taSymbol == 'ADX'):
                self.state.append(self.data['ADX'][0:self.stateLength].tolist())
        self.state.append([0])    

        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0

        return self.state

    
    def computeLowerBound(self, cash, numberOfShares, price):
        """
        GOAL: Compute the lower bound of the complete RL action space, 
              i.e. the minimum number of share to trade.
        
        INPUTS: - cash: Value of the cash owned by the agent.
                - numberOfShares: Number of shares owned by the agent.
                - price: Last price observed.
        
        OUTPUTS: - lowerBound: Lower bound of the RL action space.
        """

        # Computation of the RL action lower bound
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:  # enough cash to pay the debt
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:                # 
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound
    

    def step(self, action):
        """
        GOAL: Transition to the next trading time step based on the
              trading position decision made (either long or short).
        
        INPUTS: - action: Trading decision (1 = long, 0 = short).    
        
        OUTPUTS: - state: RL state to be returned to the RL agent.
                 - reward: RL reward to be returned to the RL agent.
                 - done: RL episode termination signal (boolean).
                 - info: Additional information returned to the RL agent.
        """

        # Seting of some local variables
        t = self.t
        tdate = self.data.index[t]
        numberOfShares = self.numberOfShares
        customReward = False

        # CASE 1: LONG POSITION, Price will go up
        if(action == 1):
            self.data.loc[tdate, 'Position'] = 1
            # Case a: Long -> Long
            if(self.data['Position'].iloc[t - 1] == 1):
                self.data.loc[tdate, 'Cash'] = self.data['Cash'].iloc[t - 1]
                self.data.loc[tdate, 'Holdings'] = self.numberOfShares * self.data['Close'].iloc[t]
            # Case b: No position -> Long
            elif(self.data['Position'].iloc[t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'].iloc[t - 1]/(self.data['Close'].iloc[t] * (1 + self.transactionCosts)))
                self.data.loc[tdate, 'Cash'] = self.data['Cash'].iloc[t - 1] - self.numberOfShares * self.data['Close'].iloc[t] * (1 + self.transactionCosts)
                self.data.loc[tdate, 'Holdings'] = self.numberOfShares * self.data['Close'].iloc[t]
                self.data.loc[tdate, 'Action'] = 1
            # Case c: Short -> Long
            else:
                tempCash = self.data['Cash'].iloc[t - 1] - self.numberOfShares * self.data['Close'].iloc[t] * (1 + self.transactionCosts)
                self.numberOfShares = math.floor(tempCash/(self.data['Close'].iloc[t] * (1 + self.transactionCosts)))
                self.data.loc[tdate, 'Cash'] = tempCash - self.numberOfShares * self.data['Close'].iloc[t] * (1 + self.transactionCosts)
                self.data.loc[tdate, 'Holdings'] = self.numberOfShares * self.data['Close'].iloc[t]
                self.data.loc[tdate, 'Action'] = 1

        # CASE 2: SHORT POSITION
        elif(action == 0):
            self.data.loc[tdate, 'Position'] = -1
            # Case a: Short -> Short
            if(self.data['Position'].iloc[t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'].iloc[t - 1], -numberOfShares, self.data['Close'].iloc[t-1])
                if lowerBound <= 0:
                    self.data.loc[tdate, 'Cash'] = self.data['Cash'].iloc[t - 1]
                    self.data.loc[tdate, 'Holdings'] =  - self.numberOfShares * self.data['Close'].iloc[t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), self.numberOfShares)
                    self.numberOfShares -= numberOfSharesToBuy
                    self.data.loc[tdate, 'Cash'] = self.data['Cash'].iloc[t - 1] - numberOfSharesToBuy * self.data['Close'].iloc[t] * (1 + self.transactionCosts)
                    self.data.loc[tdate, 'Holdings'] =  - self.numberOfShares * self.data['Close'].iloc[t]
                    customReward = True
            # Case b: No position -> Short
            elif(self.data['Position'].iloc[t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'].iloc[t - 1]/(self.data['Close'].iloc[t] * (1 + self.transactionCosts)))
                self.data.loc[tdate, 'Cash'] = self.data['Cash'].iloc[t - 1] + self.numberOfShares * self.data['Close'].iloc[t] * (1 - self.transactionCosts)
                self.data.loc[tdate, 'Holdings'] = - self.numberOfShares * self.data['Close'].iloc[t]
                self.data.loc[tdate, 'Action'] = -1
            # Case c: Long -> Short
            else:
                tempCash = self.data['Cash'].iloc[t - 1] + self.numberOfShares * self.data['Close'].iloc[t] * (1 - self.transactionCosts)
                self.numberOfShares = math.floor(tempCash/(self.data['Close'].iloc[t] * (1 + self.transactionCosts)))
                self.data.loc[tdate, 'Cash'] = tempCash + self.numberOfShares * self.data['Close'].iloc[t] * (1 - self.transactionCosts)
                self.data.loc[tdate, 'Holdings'] = - self.numberOfShares * self.data['Close'].iloc[t]
                self.data.loc[tdate, 'Action'] = -1

        # CASE 3: PROHIBITED ACTION
        else:
            raise SystemExit("Prohibited action! Action should be either 1 (long) or 0 (short).")

        # Update the total amount of money owned by the agent, as well as the return generated
        self.data.loc[tdate, 'Money'] = self.data['Holdings'].iloc[t] + self.data['Cash'].iloc[t]
        self.data.loc[tdate, 'Returns'] = (self.data['Money'].iloc[t] - self.data['Money'].iloc[t-1])/self.data['Money'].iloc[t-1]

        # Set the RL reward returned to the trading agent
        if not customReward:
            self.reward = self.data['Returns'].iloc[t]
        else:
            self.reward = (self.data['Close'].iloc[t-1] - self.data['Close'].iloc[t])/self.data['Close'].iloc[t-1]    
    
        # Same reasoning with the other action (exploration trick)
        otherAction = int(not bool(action))
        customReward = False
        if(otherAction == 1):
            otherPosition = 1
            if(self.data['Position'].iloc[t - 1] == 1):
                otherCash = self.data['Cash'].iloc[t - 1]
                otherHoldings = numberOfShares * self.data['Close'].iloc[t]
            elif(self.data['Position'].iloc[t - 1] == 0):
                numberOfShares = math.floor(self.data['Cash'].iloc[t - 1]/(self.data['Close'].iloc[t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'].iloc[t - 1] - numberOfShares * self.data['Close'].iloc[t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'].iloc[t]
            else:
                otherCash = self.data['Cash'].iloc[t - 1] - numberOfShares * self.data['Close'].iloc[t] * (1 + self.transactionCosts)
                numberOfShares = math.floor(otherCash/(self.data['Close'].iloc[t] * (1 + self.transactionCosts)))
                otherCash = otherCash - numberOfShares * self.data['Close'].iloc[t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'].iloc[t]
        else:
            otherPosition = -1
            if(self.data['Position'].iloc[t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'].iloc[t - 1], -numberOfShares, self.data['Close'].iloc[t-1])
                if lowerBound <= 0:
                    otherCash = self.data['Cash'].iloc[t - 1]
                    otherHoldings =  - numberOfShares * self.data['Close'].iloc[t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), numberOfShares)
                    numberOfShares -= numberOfSharesToBuy
                    otherCash = self.data['Cash'].iloc[t - 1] - numberOfSharesToBuy * self.data['Close'].iloc[t] * (1 + self.transactionCosts)
                    otherHoldings =  - numberOfShares * self.data['Close'].iloc[t]
                    customReward = True
            elif(self.data['Position'].iloc[t - 1] == 0):
                numberOfShares = math.floor(self.data['Cash'].iloc[t - 1]/(self.data['Close'].iloc[t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'].iloc[t - 1] + numberOfShares * self.data['Close'].iloc[t] * (1 - self.transactionCosts)
                otherHoldings = - numberOfShares * self.data['Close'].iloc[t]
            else:
                otherCash = self.data['Cash'].iloc[t - 1] + numberOfShares * self.data['Close'].iloc[t] * (1 - self.transactionCosts)
                numberOfShares = math.floor(otherCash/(self.data['Close'].iloc[t] * (1 + self.transactionCosts)))
                otherCash = otherCash + numberOfShares * self.data['Close'].iloc[t] * (1 - self.transactionCosts)
                otherHoldings = - self.numberOfShares * self.data['Close'].iloc[t]

        otherMoney = otherHoldings + otherCash

        if not customReward:
            otherReward = (otherMoney - self.data['Money'].iloc[t-1])/self.data['Money'].iloc[t-1]
        else:
            otherReward = (self.data['Close'].iloc[t-1] - self.data['Close'].iloc[t])/self.data['Close'].iloc[t-1]


     # Transition to the next trading time step
        self.t = self.t + 1

        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist()]
        for taSymbol in technicalAnalysis:
            if(taSymbol == 'MACD'):
                self.state.append(self.data['MACD'][self.t - self.stateLength : self.t].tolist())
            elif(taSymbol == 'CCI'):
                self.state.append(self.data['CCI'][self.t - self.stateLength : self.t].tolist())
            elif(taSymbol == 'RSI'):
                self.state.append(self.data['RSI'][self.t - self.stateLength : self.t].tolist())
            elif(taSymbol == 'ADX'):
                self.state.append(self.data['ADX'][self.t - self.stateLength : self.t].tolist())
        self.state.append([self.data['Position'].iloc[self.t - 1]])    

        otherState = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist()]
        for taSymbol in technicalAnalysis:
            if(taSymbol == 'MACD'):
                otherState.append(self.data['MACD'][self.t - self.stateLength : self.t].tolist())
            elif(taSymbol == 'CCI'):
                otherState.append(self.data['CCI'][self.t - self.stateLength : self.t].tolist())
            elif(taSymbol == 'RSI'):
                otherState.append(self.data['RSI'][self.t - self.stateLength : self.t].tolist())
            elif(taSymbol == 'ADX'):
                otherState.append(self.data['ADX'][self.t - self.stateLength : self.t].tolist())
        otherState.append([otherPosition])    

        if(self.t == self.data.shape[0]):
            self.done = 1  
        
        self.info = {'State' : otherState, 'Reward' : otherReward, 'Done' : self.done}

        # Return the trading environment feedback to the RL trading agent
        return self.state, self.reward, self.done, self.info


    def render(self):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the 
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.
        
        INPUTS: /   
        
        OUTPUTS: /
        """

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        self.data['Close'].plot(ax=ax1, color='blue', lw=2)
        ax1.plot(self.data.loc[self.data['Action'] == 1.0].index, 
                 self.data['Close'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax1.plot(self.data.loc[self.data['Action'] == -1.0].index, 
                 self.data['Close'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        
        # Plot the second graph -> Evolution of the trading capital
        self.data['Money'].plot(ax=ax2, color='blue', lw=2)
        ax2.plot(self.data.loc[self.data['Action'] == 1.0].index, 
                 self.data['Money'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax2.plot(self.data.loc[self.data['Action'] == -1.0].index, 
                 self.data['Money'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        
        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short"])
        ax2.legend(["Capital", "Long", "Short"])
        plt.savefig(''.join(['Figures/', str(self.marketSymbol), '_Rendering', '.png']))
        plt.close()
        #plt.show()


    def setStartingPoint(self, startingPoint):
        """
        GOAL: Setting an arbitrary starting point regarding the trading activity.
              This technique is used for better generalization of the RL agent.
        
        INPUTS: - startingPoint: Optional starting point (iteration) of the trading activity.
        
        OUTPUTS: /
        """

        # Setting a custom starting point
        self.t = np.clip(startingPoint, self.stateLength, len(self.data.index))

        # Set the RL variables common to every OpenAI gym environments    
        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist()]
        for taSymbol in technicalAnalysis:
            if(taSymbol == 'MACD'):
                self.state.append(self.data['MACD'][self.t - self.stateLength : self.t].tolist())
            elif(taSymbol == 'CCI'):
                self.state.append(self.data['CCI'][self.t - self.stateLength : self.t].tolist())
            elif(taSymbol == 'RSI'):
                self.state.append(self.data['RSI'][self.t - self.stateLength : self.t].tolist())
            elif(taSymbol == 'ADX'):
                self.state.append(self.data['ADX'][self.t - self.stateLength : self.t].tolist())
        self.state.append([self.data['Position'].iloc[self.t - 1]])    

        
        if(self.t == self.data.shape[0]):
            self.done = 1
    