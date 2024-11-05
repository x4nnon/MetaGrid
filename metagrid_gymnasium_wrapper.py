#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:20:07 2023

@author: x4nno
"""

import os

# Get the directory path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

import sys 
sys.path.append(current_dir)

import numpy as np
import Environment_obstacles
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import copy
import torch
from matplotlib import pyplot as plt
import time



class MetaGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=7, agent_location_value=2,
                goal_location_value=3, domain_size=[14,14], 
                stochastic=False, stochastic_strength=None,
                max_episode_steps=1000, style="grid", seed=0, fix_mdp=False,
                reward_gen_only=False):
        
        self.stochastic = stochastic
        self.stochastic_strength = stochastic_strength
        self.agent_location_value = agent_location_value
        self.goal_location_value = goal_location_value
        self.style = style
        self._max_episode_steps = max_episode_steps
        self.reward_gen_only = reward_gen_only
        
        self.rand_seed = seed
        
        self.env_master = Environment_obstacles.Environment(self.agent_location_value,
                                                     self.goal_location_value,
                                                     domain_size=domain_size, 
                                                     max_episode_steps=max_episode_steps,
                                                     style=self.style, seed=self.rand_seed)
        
        self.done = self.env_master.done
        self.train = True
        
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(0, 7, shape=(51,), dtype=float)
        
        # spaces.Dict(
        #     {
        #         "view": spaces.Box(0, 4, shape=(7,7), dtype=int),
        #         "dir_mag": spaces.Box(0, size, shape=(2,), dtype=float),
        #     }
        # )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.discrete.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
        # make sure these are lists
        self.action_translation = {0:('up',), 1:('down',), 2:('left',), 3:('right',)}
        
        self.fix_mdp = fix_mdp
        
    def seed(self, random_seed):
        self.rand_seed = random_seed
        self.env_master.seed(random_seed)
        
    def _get_obs(self):
        view, dir_mag = self.env_master.get_observation_space()
        view = view.flatten()
        return np.concatenate((view, dir_mag))
        # return {"view": view, "dir_mag": dir_mag}
    
    def get_obs(self):
        view, dir_mag = self.env_master.get_observation_space()
        view = view.flatten()
        return np.concatenate((view, dir_mag))
        # return {"view": view, "dir_mag": dir_mag}
    
    def _get_info(self):
        return {"info": "Info not implemented"}
    
    def get_info(self):
        return {"info": "Info not implemented"}
    
    def reset(self, goal_choice=[], start_same=False, total_random=False, seed=None, options=None, train=True):
        if self.fix_mdp:
            start_same=True
            if not start_same:
                self.seed(self.rand_seed)
            
        elif self.reward_gen_only:
            total_random=True

        # elif seed:
        #     self.seed(seed)
            
        self.episode_start_times = np.full(
            1, time.perf_counter(), dtype=np.float32
        )
        
        self.episode_returns = np.zeros(1, dtype=np.float32)
        self.episode_lengths = np.zeros(1, dtype=np.int32)
        
        self.env_master.reset(goal_choice=goal_choice,
                       start_same=start_same, 
                       total_random=total_random,
                       train=self.train, seed=seed)
        
        self.done = self.env_master.done
        self._elapsed_steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def render(self):
        plt.imshow(self.env_master.domain)
        plt.show()
        plt.close()
    
    def step(self, action):
        
        if not isinstance(action, int):
            action = int(action) # take from tensor.
            
        action = self.action_translation[action]
        domain, reward, terminated, info = self.env_master.step(action)
        info = {"not implemented":info}
        
        
        if terminated == True:
            obs_space = copy.deepcopy(self.env_master.domain)
            obs_space = np.pad(obs_space, ((3,3),(3,3)), "constant", constant_values=((1,1),(1,1)))
            obs_centre = Environment_obstacles.find_agent_location(obs_space)
            obs_space_1 = obs_space[obs_centre[0]-3:obs_centre[0]+4, obs_centre[1]-3:obs_centre[1]+4]
            obs_space_1[3][3] = 0
            
            if np.where(obs_space_1==3)[0].size != 0:
                goal_x = np.where(obs_space_1==3)[0][0]
                goal_y = np.where(obs_space_1==3)[1][0]
                obs_space_1[goal_x][goal_y] = 0
                    
            # also remove the wall just because the VAE works better like this.
            if np.where(obs_space_1==4)[0].size != 0:
                goal_x = np.where(obs_space_1==4)[0][0]
                goal_y = np.where(obs_space_1==4)[1][0]
                obs_space_1[goal_x][goal_y] = 0
            
            observation = [obs_space_1, torch.tensor([0, 0])]
            
            view, dir_mag = observation
            view = view.flatten()
            observation = np.concatenate((view, dir_mag))
            
        else:
            observation = self._get_obs()
            
        
        self.done = self.env_master.done
        terminated = self.done
        
        self._elapsed_steps += 1
        
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        else:
            truncated = None
            
        self.episode_returns += reward
        self.episode_lengths += 1
        dones = np.logical_or(terminated, truncated)
        num_dones = np.sum(dones)
        # if num_dones:
        #     if "episode" in info or "_episode" in info:
        #         raise ValueError(
        #             "Attempted to add episode stats when they already exist"
        #         )
        #     else:
        #         info["episode"] = {
        #             "r": np.where(dones, self.episode_returns, 0.0),
        #             "l": np.where(dones, self.episode_lengths, 0),
        #             "t": np.where(
        #                 dones,
        #                 np.round(time.perf_counter() - self.episode_start_times, 6),
        #                 0.0,
        #             ),
        #         }

            # self.return_queue.extend(self.episode_returns[dones])
            # self.length_queue.extend(self.episode_lengths[dones])
            # self.episode_count += num_dones
            # self.episode_lengths[dones] = 0
            # self.episode_returns[dones] = 0
            # self.episode_start_times[dones] = time.perf_counter()
        
        # debug only
        # plt.imshow(self.env_master.domain)
        # plt.title(f"{action}")
        # plt.show()
        
        return observation, reward, terminated, truncated, info
    
    
    def fracos_step(self, action, next_ob, agent, total_rewards=0, total_steps_taken=0):
        """This needs to manage recursively taking actions"""
        # !!! Need to add in tracking of the steps -- unless this is done in the infos.
        try:
            ob = tuple(next_ob.cpu().numpy())
        except:
            ob = tuple(next_ob)
        
        if action not in range(agent.action_prims):
            if ob not in agent.discrete_search_cache.keys():
                agent.initial_search(ob)
            id_actions = tuple(agent.discrete_search_cache[ob][action])
            if isinstance(id_actions[0], np.ndarray):
                for id_action in id_actions:
                    for reverse_cypher in agent.reverse_cyphers:
                        if tuple(id_action) in reverse_cypher.keys():
                            id_action = reverse_cypher[tuple(id_action)]
                            break
    
                    next_ob, total_rewards, termination, truncation, info, total_steps_taken = \
                        self.fracos_step(id_action, next_ob, agent, total_rewards=total_rewards, total_steps_taken=total_steps_taken)
                        
                    # need to exit if we have finished
                    next_done = np.logical_or(termination, truncation)
                    if next_done:
                        return next_ob, total_rewards, termination, truncation, info, total_steps_taken
                        
            else:
                # returns a negative reward and our current location.
                return ob, -0.1, False, None, {"not implemented" : []}, total_steps_taken
        else:
            next_ob, reward, termination, truncation, info = self.step(action)
            total_rewards += reward
            total_steps_taken += 1
            next_done = np.logical_or(termination, truncation)
            if next_done:
                return next_ob, total_rewards, termination, truncation, info, total_steps_taken
            
        return next_ob, total_rewards, termination, truncation, info, total_steps_taken
        
    # def render(self):
    #     if self.render_mode == "rgb_array":
    #         return "rendering has not been implemented yet"
        
    def close(self):
        pass
    
if __name__ == "__main__":
    
    register( id="x4nno/metagrid-v0",
             entry_point="metagrid_gymnasium_wrapper:MetaGridEnv",
             max_episode_steps=500,)
    
    env = gym.make("x4nno/metagrid-v0")
