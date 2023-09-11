import os
import pickle
import socket
import time

import numpy as np

from objects import InteractiveObj, RBOObject, YCBObject
from panda_env2 import Panda
import pybullet as p
import pybullet_data
from tf import *


class CommClass(object):
    """Parent class of CommClient, CommServer. Connects to a port via socket
    library for advertising / listening"""

    def __init__(self, ip, port=8000):
        """
        Initializes the CommClass. Note that the initialization may hang as the
        object waits for a connection via `self.connect2comms`.

        Parameters:
        ----------
        `ip` : `string`
            ip to connect to. note that you should use "127.0.0.1" for the
            loopback device
        `port` : `int`
            port to use for socket communication
        """
        self.ip = ip
        self.port = port
        print("awaiting...")
        conn, addr = self.connect2comms()
        print("connected!")
        self.conn = conn
        self.addr = addr

    def connect2comms(self):
        """
        Connects to a socket via `socket` library.

        Returns
        -------
        `conn` : `socket.connection`
        `addr` : `socket.address`
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.ip, self.port))
        s.listen()
        conn, addr = s.accept()
        return conn, addr


class CommServer(CommClass):
    """This child of CommClass can send comms, but cannot receive."""

    def send2comms(self, msg):
        """sends an arbitrary message via socket

        Parameters
        ----------
        `msg` : `any`
            the message to be sent. must be able to be serialized by
            `pickle.dumps(msg)`

        Returns
        -------
        None
        """
        send_msg = pickle.dumps(msg)
        self.conn.send(send_msg)
        return

    def send(self, *args):
        """Wrapper around `send2comms`"""
        return self.send2comms(*args)

    def __call__(self, *args):
        """Wrapper around `send`"""
        return self.send(*args)


class CommClient(CommClass):
    """Child class of CommClass, can only listen to comms, cannot send."""

    def __init__(self, ip, port):
        """Initializes CommClient object. Note that this initialization may hang
        as the socket connection is estabilished. As opposed to the parent
        class `CommClass`, `self.conn` and `self.addr` are not initialized
        here

        Parameters:
        ----------
        `ip` : `string`
            string to use for ip. note that "127.0.0.1" should be used for loopback device
        `port` : `int`
            port to use for communication
        """
        self.ip = ip
        self.port = port
        print("awaiting...")
        self.s = self.connect2comms()
        print("connected!")
        self.conn = None
        self.addr = None

    def connect2comms(self):
        """Helper method to establish socket communication

        Returns:
        -------
        `s` : `socket.socket`
            socket to use for communication
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.connect((self.ip, self.port))
        return s

    def listen2comms(self):
        """Listens to socket, awaits communication synchronously.

        Returns:
        --------
        `recv` : `any`
            pickle deserialized data
        """
        message = self.s.recv(2048)
        recv = pickle.loads(message)
        return recv

    def __call__(self):
        return self.listen2comms()


class SimpleEnv:
    """
    A simple pybullet environment to use for visualization and sim.
    Note that this is flipped 180 deg from the real world setup:
    multiply (x, y) by -1 to mimic the visualization of the real world.
    """

    def __init__(self, visualize=False):
        self.urdfRootPath = pybullet_data.getDataPath()
        print(self.urdfRootPath)
        if visualize:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)

        self._set_camera()
        p.loadURDF(
            os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65]
        )
        p.loadURDF(
            os.path.join(self.urdfRootPath, "table/table.urdf"),
            basePosition=[0.5, 0, -0.65],
        )

        self.fork = RBOObject("fork_edit")
        self.fork.load()
        self.fork_position = [0.5, 0.3, 0.02]

        self.fork_quaternion = [0.0, 0.0, 0.70710678, 0.70710678]
        self.fork_poslist = [0.5, 0.3, 0.02]
        self.fork_quatlist = [1.0, 0.0, 0.0, 0.0]
        self.fork_grasp = [0]
        self.fork_details = {
            "obj": self.fork,
            "grasp": self.fork_grasp,
            "positions": self.fork_poslist,
            "quats": self.fork_quatlist,
            "num": len(self.fork_grasp),
        }
        self.fork.set_position_orientation(self.fork_position, self.fork_quaternion)
        self.panda = Panda()

    def reset_box(self):
        self.block.set_position_orientation(self.block_position, self.block_quaternion)

    def reset(
        self, q=[0.0, -np.pi / 4, 0.0, -2 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
    ):
        self.panda.reset(q)
        return [self.panda.state]

    def close(self):
        p.disconnect()

    def state(self):
        return self.panda.state

    def step(self, joint=[0] * 7, pos=[0] * 3, quat=[0] * 4, grasp=True, mode=1):
        state = self.panda.state
        self.panda.step(
            mode=mode, djoint=joint, dposition=pos, dquaternion=quat, grasp_open=grasp
        )

        p.stepSimulation()
        time.sleep(0.01)

        next_state = self.panda.state
        reward = 0.0
        done = False
        info = {}
        return next_state, reward, done, info

    def render(self):
        (width, height, pxl, depth, segmentation) = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
        )
        rgb_array = np.array(pxl, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=30,
            cameraPitch=-60,
            cameraTargetPosition=[0.5, -0.2, 0.0],
        )
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0, 0],
            distance=1.0,
            yaw=90,
            pitch=-50,
            roll=0,
            upAxisIndex=2,
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.camera_width) / self.camera_height,
            nearVal=0.1,
            farVal=100.0,
        )
