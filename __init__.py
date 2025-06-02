#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:38:51 2023

@author: x4nno
"""

print("pre_imports")

from gymnasium.envs.registration import register 
from .metagrid_gymnasium_wrapper import MetaGridEnv

print("pre_register")

# Register the environment
# The id should be in the format "namespace/name-version"
# The entry_point should be "package_name.module_name:ClassName"
register(
    id="MetaGridEnv/metagrid-v0", # This ID seems fine.
    entry_point="MetaGridEnv.metagrid_gymnasium_wrapper:MetaGridEnv", # Corrected
    max_episode_steps=500,
)

print("MetaGridEnv/metagrid-v0 registered with Gymnasium via __init__.py") # For debugging