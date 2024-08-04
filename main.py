# coding=utf-8

"""
Goal: Program Main.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import argparse

from tradingSimulator import TradingSimulator


###############################################################################
##################################### MAIN ####################################
###############################################################################

if(__name__ == '__main__'):

    # Retrieve the paramaters sent by the user
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-strategy", default='TDQN', type=str, help="Name of the trading strategy")
    parser.add_argument("-stock", default='Tesla', type=str, help="Name of the stock (market)")
    args = parser.parse_args()
    
    # Initialization of the required variables
    simulator = TradingSimulator()
    strategy = args.strategy
    stock = args.stock

    # Training and testing of the trading strategy specified for the stock (market) specified
    simulator.simulateNewStrategy(strategy, stock, saveStrategy=True, retraining=False)
    
#    simulator.simulateExistingStrategy(strategy, stock, showAll=True)          # plot chart
#    simulator.evaluateStock(stock)
    
    
    #simulator.displayTestbench()
    #simulator.analyseTimeSeries(stock)
    #simulator.simulateNewStrategy(strategy, stock, saveStrategy=False)
    #simulator.simulateExistingStrategy(strategy, stock)          # plot chart
    #simulator.evaluateStrategy(strategy, saveStrategy=False)     
    #simulator.evaluateStock(stock)                               # plot table
   