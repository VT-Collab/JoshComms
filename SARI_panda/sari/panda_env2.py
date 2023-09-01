import os
import numpy as np
import pybullet as p
import pybullet_data
from utils_panda import joint2pose,get_quaternion_from_euler

class Panda():

    def __init__(self, basePosition=[0,0,0], visualize=False):
        self.urdfRootPath = pybullet_data.getDataPath()
        
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        #p.setGravity(0, 0, -9.81)
        # set up camera
        p.setGravity(0, 0, -9.81)
        self._set_camera()
        # load some scene objects
        p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        self.panda = p.loadURDF(os.path.join(self.urdfRootPath,"franka_panda/panda.urdf"),useFixedBase=True,basePosition=basePosition)
        self.reset()
        #self.limit_low = [-2.60752, -1.58650, -2.60752, -2.764601, -2.607521, -0.015707, -2.60752]
        #self.limit_high = [2.60752, 1.58650, 2.60752, -0.062831, 2.60752, 3.37721, 2.60752]
        

    """functions that environment should use"""

    # has two modes: joint space control (0) and ee-space control (1)
    # djoint is a 7-dimensional vector of joint velocities
    # dposition is a 3-dimensional vector of end-effector linear velocities
    # dquaternion is a 4-dimensional vector of end-effector quaternion velocities
    def step(self, mode=0, djoint=[0]*7, dposition=[0]*3, dquaternion=[0]*4, grasp_open=True):

        # velocity control
        self._velocity_control(mode=mode, djoint=djoint, dposition=dposition, dquaternion=dquaternion, grasp_open=grasp_open)

        # update robot state measurement
        self.read_state()
        self.read_jacobian()


    def reset(self):
        init_pos = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.05, 0.05]
        self._reset_robot(init_pos)


    """internal functions"""

    def read_state(self):
        joint_position = [0]*9
        joint_velocity = [0]*9
        joint_torque = [0]*9
        joint_states = p.getJointStates(self.panda, range(9))
        for idx in range(9):
            joint_position[idx] = joint_states[idx][0]
            joint_velocity[idx] = joint_states[idx][1]
            joint_torque[idx] = joint_states[idx][3]
        ee_states = p.getLinkState(self.panda, 11)
        ee_position = list(ee_states[4])
        ee_quaternion = list(ee_states[5])
        gripper_contact = p.getContactPoints(bodyA=self.panda, linkIndexA=10)
        self.state = {}
        self.state['q'] = np.asarray(joint_position)
        self.state['joint_velocity'] = np.asarray(joint_velocity)
        self.state['joint_torque'] = np.asarray(joint_torque)
        self.state['ee_position'] = np.asarray(ee_position)
        self.state['ee_quaternion'] = np.asarray(ee_quaternion)
        self.state['ee_euler'] = np.asarray(p.getEulerFromQuaternion(ee_quaternion))
        self.state['gripper_contact'] = len(gripper_contact) > 0
        # get cartesian pose
        xyz_lin, R = joint2pose(joint_position)
        #print(xyz_lin)
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        xyz_ang = [alpha, beta, gamma]
        xyz = np.asarray(xyz_lin).tolist() + np.asarray(xyz_ang).tolist()
        self.state["x"] = np.array(xyz)
   # return 

    def read_jacobian(self):
        linear_jacobian, angular_jacobian = p.calculateJacobian(self.panda, 11, [0, 0, 0], list(self.state['q']), [0]*9, [0]*9)
        linear_jacobian = np.asarray(linear_jacobian)[:,:7]
        angular_jacobian = np.asarray(angular_jacobian)[:,:7]
        full_jacobian = np.zeros((6,7))
        full_jacobian[0:3,:] = linear_jacobian
        full_jacobian[3:6,:] = angular_jacobian
        self.state['J'] = full_jacobian
        self.state['linear_jacobian'] = linear_jacobian
        self.state['angular_jacobian'] = angular_jacobian

    def _reset_robot(self, joint_position):
        self.state = {}
        self.jacobian = {}
        self.desired = {}
        for idx in range(len(joint_position)):
            p.resetJointState(self.panda, idx, joint_position[idx])
        self.read_state()
        self.read_jacobian()
        self.desired['joint_position'] = self.state['q']
        self.desired['ee_position'] = self.state['ee_position']
        self.desired['ee_quaternion'] = self.state['ee_quaternion']

    def _inverse_kinematics(self, ee_position, ee_quaternion):
        return p.calculateInverseKinematics(self.panda, 11, list(ee_position), list(ee_quaternion))

    def _velocity_control(self, mode, djoint, dposition, dquaternion, grasp_open):
        if mode:
            self.desired['ee_position'] += np.asarray(dposition) / 240.0
            self.desired['ee_quaternion'] += np.asarray(dquaternion) / 240.0
            q_dot = self._inverse_kinematics(self.desired['ee_position'], self.desired['ee_quaternion']) - self.state['q']
        else:
            self.desired['joint_position'] += np.asarray(list(djoint)+[0, 0]) / 240.0
            q_dot = self.desired['joint_position'] - self.state['q']
        gripper_position = [0.0, 0.0]
        if grasp_open:
            gripper_position = [0.05, 0.05]
        p.setJointMotorControlArray(self.panda, range(9), p.VELOCITY_CONTROL, targetVelocities=list(q_dot))
        p.setJointMotorControlArray(self.panda, [9,10], p.POSITION_CONTROL, targetPositions=gripper_position)
    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=30, cameraPitch=-60,
                                     cameraTargetPosition=[0.5, -0.2, 0.0])
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                               distance=1.0,
                                                               yaw=90,
                                                               pitch=-50,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.camera_width) / self.camera_height,
                                                        nearVal=0.1,
                                                        farVal=100.0)
    def pose2joint(self,pose):
        quat = get_quaternion_from_euler(pose[3], pose[4], pose[5])
        pos = pose[:3]
        # if guess is None:
        #     guess = self.joint_states
        q = self._inverse_kinematics( pos, quat)
        return q
