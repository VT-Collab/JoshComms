import numpy as np
import scipy.misc
import time
from Utils import *  
#import IPython

logsumexp = scipy.special.logsumexp

class GoalPredictor(object):
    max_prob_any_goal = 0.99
    log_max_prob_any_goal = np.log(max_prob_any_goal)

    def __init__(self, goals):
        self.goals = goals
        self.in_control = False
        self.movement_timer = time.time()
        self.log_goal_distribution = (np.ones(len(self.goals))/np.linalg.norm(np.ones(len(self.goals)))) #scalar
        #print(self.log_goal_distribution)


#   def update_distribution(self, values, q_values,beta=0.5):
#     self.log_goal_distribution *=  np.exp(beta*(values-q_values))/np.average(np.exp(beta*(values-q_values)))
#     # if max(self.log_goal_distribution-(np.sort(self.log_goal_distribution)[-2])) > .3:
#     #   print(max(self.log_goal_distribution))
#     #   print(np.sort(self.log_goal_distribution)[-2])
#     #   time.sleep(5)
#     #print(np.exp(beta*(values-q_values)))
#     self.normalize_log_distribution()
#     print(self.log_goal_distribution)
#     #self.clip_prob()

    def update_distribution(self, values, q_values,user_action,robot_state,beta=0.5,threshold = .1):
        self.user_action = user_action
        self.robot_state = robot_state
        goal_distribution = self.get_distribution()
        max_prob_goal_ind = np.argmax(goal_distribution)
        if np.linalg.norm(self.user_action) < threshold:
            temp_timer = time.time()
            if (temp_timer - self.movement_timer)>1:
                if self.in_control == False:
                    self.in_control = True
                    self.log_goal_distribution = np.zeros(np.shape(self.log_goal_distribution))
                self.log_goal_distribution[max_prob_goal_ind] = 1.0
                #print("semi")
        else:
            if self.in_control == True:
                self.in_control = False
                self.log_goal_distribution = (np.ones(len(self.goals))/np.linalg.norm(np.ones(len(self.goals))))
            self.log_goal_distribution *=  np.exp(beta*(values-q_values))/np.average(np.exp(beta*(values-q_values)))
        
        self.normalize_log_distribution()          

    def normalize_log_distribution(self):
        log_normalization_val = np.linalg.norm(self.log_goal_distribution)
       # print("ERROR SPOT ______")
        #print(log_normalization_val)
       # print(self.log_goal_distribution)
        self.log_goal_distribution /= log_normalization_val
        #self.clip_prob()

    def clip_prob(self):
        if len(self.log_goal_distribution) <= 1:
            return
        #check if any too high
        max_prob_ind = np.argmax(self.log_goal_distribution)
        if self.log_goal_distribution[max_prob_ind] > self.log_max_prob_any_goal:
            #see how much we will remove from probability
            diff = np.exp(self.log_goal_distribution[max_prob_ind]) - self.max_prob_any_goal
            #want to distribute this evenly among other goals
            diff_per = diff/(len(self.log_goal_distribution)-1.)

            #distribute this evenly in the probability space...this corresponds to doing so in log space
            # e^x_new = e^x_old + diff_per, and this is formulate for log addition
            self.log_goal_distribution += np.log( 1. + diff_per/np.exp(self.log_goal_distribution))
            #set old one
            self.log_goal_distribution[max_prob_ind] = self.log_max_prob_any_goal

    def get_distribution(self):
        return (self.log_goal_distribution)
