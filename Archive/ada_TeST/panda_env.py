import numpy as np
import pybullet as p
import pybullet_data
import os
from Utils import *

class Panda():

    def __init__(self, basePosition=[0,0,0]):
        self.urdfRootPath = pybullet_data.getDataPath()
        self.panda = p.loadURDF(os.path.join(self.urdfRootPath,"franka_panda/panda.urdf"),
                useFixedBase=True, basePosition=basePosition)
    

    def reset(self):
        init_pos = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.05, 0.05]
        for idx in range(len(init_pos)):
            p.resetJointState(self.panda, idx, init_pos[idx])        

    def reset_task(self, ee_position, ee_quaternion):
        self.reset()
        init_pos = self.inverse_kinematics(ee_position, ee_quaternion)
        for idx in range(len(init_pos)):
            p.resetJointState(self.panda, idx, init_pos[idx])

    def reset_joint(self, joint_position):
        init_pos = list(joint_position) + [0.0, 0.0, 0.05, 0.05]
        for idx in range(len(init_pos)):
            p.resetJointState(self.panda, idx, init_pos[idx])

    def read_state(self):
        joint_position = [0]*9
        joint_states = p.getJointStates(self.panda, range(9))
        for idx in range(9):
            joint_position[idx] = joint_states[idx][0]
        ee_states = p.getLinkState(self.panda, 11)
        ee_position = list(ee_states[4])
        ee_quaternion = list(ee_states[5])
        gripper_contact = p.getContactPoints(bodyA=self.panda, linkIndexA=10)
        self.state = {}
        self.state['q'] = np.array(joint_position)
        self.state['ee_position'] = np.array(ee_position)
        self.state['ee_quaternion'] = np.array(ee_quaternion)
        self.state['gripper_contact'] = len(gripper_contact) > 0

        # get cartesian pose
        xyz_lin, R = joint2pose(joint_position)
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        xyz_ang = [alpha, beta, gamma]
        xyz = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
        self.state["x"] = np.array(xyz)
        
        linear_jacobian, angular_jacobian = p.calculateJacobian(self.panda, 11, [0, 0, 0], list(joint_position), [0]*9, [0]*9)
        linear_jacobian = np.asarray(linear_jacobian)[:,:7]
        angular_jacobian = np.asarray(angular_jacobian)[:,:7]
        full_jacobian = np.zeros((6,7))
        full_jacobian[0:3,:] = linear_jacobian
        full_jacobian[3:6,:] = angular_jacobian
        self.state['J'] = full_jacobian
        return self.state

    def _inverse_kinematics(self, ee_position, ee_quaternion):
        return p.calculateInverseKinematics(self.panda, 11, list(ee_position), list(ee_quaternion), maxNumIterations=5)

    def traj_task(self, traj, time):
        state = self.read_state()
        pd = traj.get_waypoint(time)
        qd = self.inverse_kinematics(pd, [1, 0, 0, 0])
        q_dot = 100 * (qd - state["q"])
        p.setJointMotorControlArray(self.panda, range(9), p.VELOCITY_CONTROL, targetVelocities=list(q_dot))
        _ = self.read_state()

    def traj_q(self, qd):
        state = self.read_state()
        q = state["q"]
        q_dot = 100 * (qd - q[:7])
        #print("DEATH_TO_THE_FALSE_GOD",qd)
        q_dot = np.append(qd,[0,0])
        p.setJointMotorControlArray(self.panda, range(9), p.VELOCITY_CONTROL, targetVelocities=list(q_dot))

    def traj_joint(self, traj, time):
        state = self.read_state()
        qd = traj.get_waypoint(time)
        q_dot = 100 * (qd - state["q"])
        p.setJointMotorControlArray(self.panda, range(9), p.VELOCITY_CONTROL, targetVelocities=list(q_dot))        