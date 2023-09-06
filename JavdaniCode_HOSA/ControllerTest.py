import numpy as np
import pygame
import math
import time
import os, sys


class Joystick(object):

    def __init__(self):
        pygame.init()
        pygame.display.set_caption('game base')
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.deadband = 0.1
        self.timeband = 0.5
        self.lastpress = time.time()

    def input(self):
        pygame.event.get()
        curr_time = time.time()
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        z3 = self.gamepad.get_axis(4)
        if abs(z1) < self.deadband:
            z1 = 0.0
        if abs(z2) < self.deadband:
            z2 = 0.0
        if abs(z3) < self.deadband:
            z3 = 0.0
        A_pressed = self.gamepad.get_button(0) 
        B_pressed = self.gamepad.get_button(1) 
        X_pressed = self.gamepad.get_button(2) 
        Y_pressed = self.gamepad.get_button(3) 
        START_pressed = self.gamepad.get_button(7) 
        STOP_pressed = self.gamepad.get_button(6) 
        Right_trigger = self.gamepad.get_button(5)
        Left_Trigger = self.gamepad.get_button(4)
        if A_pressed or START_pressed or B_pressed:
            self.lastpress = curr_time
        return [z1, z2, z3], A_pressed, B_pressed, X_pressed, Y_pressed, START_pressed, STOP_pressed, Right_trigger, Left_Trigger
    def rumble(self,time):
        self.gamepad.rumble(0,1,100)
    

def main():
    stick = Joystick()
    
    while True:
        [z1, z2, z3], A_pressed, B_pressed, X_pressed, Y_pressed, START_pressed, STOP_pressed, Right_trigger, Left_Trigger = stick.input()
        stick.gamepad.rumble(0,1,0)
        print("Cycle")
        if A_pressed == True:
            print("AAAAAAAAAAAh")
        time.sleep(1)
        stick.gamepad.stop_rumble()
        time.sleep(1)

main()