#######################################
# GameTrainer v0.0 ####################
#######################################

############################
# Imports ##################
############################

import gym 
import numpy as np
import random
from IPython.display import clear_output
import math 
import time
from sklearn.preprocessing import KBinsDiscretizer

############################
# GameTrainer Class ########
############################

class GameTrainer:
    def __init__(self, alpha = 0.1, gamma = 0.6, epsilon = 0.1):
        
        self.env = gym.make('Taxi-v3').env

        
        self.Qmatrix = self.initQMatrix(self.getActionDomain(), self.getStatesDomain())
        
        #hyperparams
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        
        #Storing for stats
        self.all_epochs = []
        self.all_penalties = []
        
        
    def getActionDomain(self):
        return self.env.action_space.n
        
    def getStatesDomain(self):
        return self.env.observation_space.n
    
    def initQMatrix(self, nActions, nStates):
        Qmatrix = np.zeros([nStates, nActions])
        return Qmatrix
        
    def train(self, n_epochs = 25000, render = True):
        for i in range(1, n_epochs+1):
            state = self.env.reset()
            
            epochs, penalties, reward = 0, 0, 0
            done = False
            
            while not done: 
                if(random.uniform(0, 1) < self.epsilon):
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Qmatrix[state])
                    
                next_state, reward, done, info = self.env.step(action)
                
                old_value = self.Qmatrix[state, action]
                next_max = np.max(self.Qmatrix[next_state])
                new_value = (1-self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.Qmatrix[state, action] = new_value
                
                if reward == -10:
                    penalties +=1
                    
                state = next_state
                epochs += 1
            
            if(i % 100 == 0):
                clear_output(wait = True)
                print(f"Episode: {i}")
                    
        print("Q-learning train ended after ", n_epochs, " epochs.")
        
        
    def test(self, nTests = 1):
        for i in range(1, nTests+1):
            state = self.env.reset()

            done = False
            while not done: 
                time.sleep(0.5)
                action = np.argmax(self.Qmatrix[state])
                next_state, reward, done, info = self.env.step(action)
                state = next_state 
                self.env.render()
                
        print('Done !')
        
    def displayRandomGameCartTaxi(self, nTests = 1):
        env = gym.make('Taxi-v3')
        env.reset()
        for _ in range(50):
            env.render()
            time.sleep(0.5)
            env.step(env.action_space.sample()) # take a random action
        env.close()
                
    #def displayPolicyGame(self):
        
    

############################
# Code Example Usage #######
############################

gt = GameTrainer()
#gt.displayRandomGameCartTaxi()
gt.train(n_epochs = 10000, render = False)
gt.test()









