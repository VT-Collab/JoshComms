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

    def update_distribution(self, values, q_values,user_action,robot_state,beta=0.5):
        self.user_action = user_action
        self.robot_state = robot_state
        self.log_goal_distribution *=  np.exp(beta*(values-q_values))/np.average(np.exp(beta*(values-q_values)))
        self.comm_prior()
        self.normalize_log_distribution()

    def comm_prior(self,threshold = .1):
        #if they haven't moved for a little bit
        goal_distribution = self.get_distribution()
        max_prob_goal_ind = np.argmax(goal_distribution)
        
        if np.linalg.norm(self.user_action) < threshold:
            temp_timer = time.time()
            if (temp_timer - self.movement_timer)>1:
                self.log_goal_distribution[max_prob_goal_ind] =1
                print("semi")
        else:
            #print(max_prob_goal_ind,"TEEEST")
            use = (((self.goals[max_prob_goal_ind]).pos)-(self.robot_state["x"])[0:3])
            use = use/np.linalg.norm(use)
            moved_joints = (self.robot_state["q"])[0:7] + self.user_action
            pos, H = joint2pose(moved_joints)
            use2 = (pos -(self.robot_state["x"])[0:3])
            use2 = use2/np.linalg.norm(use2)        
            self.log_goal_distribution[max_prob_goal_ind] *= 1.1*np.exp(-np.linalg.norm(use2-use))
            #print("Functional",1.1*np.exp(-np.linalg.norm(use2-use)))
        self.movement_timer = time.time()



        
        #If they have moved, was it in a 15 cone of the confident object
        #Blank = most confident sorted object
        #Normalized Coordinate vector to Blank
        #Difference between action vector and coordinate vector.
        
            
            

    def normalize_log_distribution(self):
        log_normalization_val = np.linalg.norm(self.log_goal_distribution)
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
