import numpy as np
import scipy

#import tf
import tf as transmethods

from Utils import *

# Goals for prediction and assistance
# pose corresponds to the pose of the goal object
# target poses corresponds to all the grasp locations for this object

class Goal: 
    
    def __init__(self, pose,pos, grasp,target_poses = list(), target_iks = list(),priority = [True,True]):
      self.pose = pose
      self.quat = pose
      self.goal_num = 0
      self.pos = list(pos)
      self.grasp= grasp
      if not target_poses:
        target_poses.append(pose)

      #copy the targets
      self.target_poses = list(target_poses)
      self.target_iks = list(target_iks)
      self.target_quaternions = self.quat
      self.priority = priority
      #self.compute_quaternions_from_target_poses()

      #print 'NUM POSES: ' + str(len(self.target_poses))

    # def compute_quaternions_from_target_poses(self):
    #   #print(self.target_poses)
    #   self.target_quaternions = [transmethods.quaternion_from_matrix(target_pose) for target_pose in self.target_poses]
    
    def at_goal(self, end_effector_trans):
      for pose,quat in zip(self.target_poses,self.target_quaternions):
        pos_diff =  self.pos - end_effector_trans[0:3,3]
        trans_dist = np.linalg.norm(pos_diff)
        
        #print("Quat",quat)
        quat_dist = QuaternionDistance(transmethods.quaternion_from_matrix(end_effector_trans), quat)

        if (trans_dist < 0.01) and (quat_dist < np.pi/48):
          return True
      # if none of the poses in target_poses returned, then we are not at goal
      return False
    def update(self):
      if self.goal_num + 1 <= len(self.grasp):
        self.goal_num += 1
      else:
        print("Object Task Complete") #possible addition to the user face saying robot thinks task is done and is no longer confident in helping
    def basiba():
      print("Bamp Basiba")
