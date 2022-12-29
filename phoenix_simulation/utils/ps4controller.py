"""
Copyright (c) 2022 Sven Gronauer (Technical University of Munich)

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file presents an interface for interacting with the Playstation 4
# Controller in Python. Simply plug your PS4 controller into your computer using
# USB and run this script!
#
# NOTE: I assume in this script that the only joystick plugged in is the PS4
# controller. if this is not the case, you will need to change the class
# accordingly.
#
# Copyright © 2015 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

import os
import pprint
import pygame
import time


class PS4Controller(object):
    """Class representing the PS4 controller. Pretty straightforward
    functionality."""
    controller = None
    axis_data = None
    button_data = None
    hat_data = None

    def __init__(self):
        """Initialize the joystick components"""

        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

        # if not self.axis_data:
        self.axis_data = {}

        # if not self.button_data:
        self.button_data = {}
        for i in range(self.controller.get_numbuttons()):
            self.button_data[i] = False

        # if not self.hat_data:
        self.hat_data = {}
        for i in range(self.controller.get_numhats()):
            self.hat_data[i] = (0, 0)

    def fetch_inputs(self):
        """Update internals through listing for events to happen..."""
        try:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    self.axis_data[event.axis] = round(event.value, 3)
                elif event.type == pygame.JOYBUTTONDOWN:
                    self.button_data[event.button] = True
                elif event.type == pygame.JOYBUTTONUP:
                    self.button_data[event.button] = False
                elif event.type == pygame.JOYHATMOTION:
                    self.hat_data[event.hat] = event.value
        except KeyError:
            print(f'Call PS4Controller.fetch_inputs() before reading inputs.')


if __name__ == "__main__":
    ps4 = PS4Controller()
    ps4.fetch_inputs()
