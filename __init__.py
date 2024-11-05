#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:38:51 2023

@author: x4nno
"""

import sys, os

from gymnasium.envs.registration import register 

from .metagrid_gymnasium_wrapper import MetaGridEnv

sys.path.append('/home/x4nno/Documents/PhD/MetaGridEnv/MetaGridEnv')

register( id="MetaGridEnv/metagrid-v0",
         entry_point="metagrid_gymnasium_wrapper:MetaGridEnv",
         max_episode_steps=500,)
