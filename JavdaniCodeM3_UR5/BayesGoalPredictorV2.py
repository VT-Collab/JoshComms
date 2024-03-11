import numpy as np
import scipy.misc
#import IPython

logsumexp = scipy.special.logsumexp

class GoalPredictor(object):

  
  def __init__(self, goals):
    self.goals = goals
    self.min_prob = .075
    self.count = 0

    self.log_goal_distribution = ((1./len(self.goals))*np.ones(len(self.goals))) #scalar

    #print( "STARTING --- [Can,Mug]",self.log_goal_distribution)

  def update_distribution(self,  BaseQVal,qvalues,values,beta = .05):
    #Here bring in z for qv
    self.count+=1
    
    self.log_goal_distribution *= np.exp(beta*-1*( (qvalues) + BaseQVal))
   
    self.normalize_log_distribution()


  def normalize_log_distribution(self):
    if min(self.log_goal_distribution) < self.min_prob:
      induse = np.where(self.log_goal_distribution == min(self.log_goal_distribution))[0][0]
      self.log_goal_distribution[induse] = self.min_prob
    log_normalization_val = np.linalg.norm(self.log_goal_distribution)
    self.log_goal_distribution /= log_normalization_val


  def get_distribution(self):
    return (self.log_goal_distribution)
