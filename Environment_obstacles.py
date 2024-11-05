# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 14:57:38 2021

@author: x4nno
"""

import domain_creation_obstacles
import building_blocks_obstacles
import random
import numpy as np
import copy
from matplotlib import pyplot as plt
import matplotlib
import torch
import time

NEGATIVE_REWARD = -1
STEP_REWARD = -0.001 # change this as you need to and then pip install .
POSITIVE_REWARD = 1
WALL_PUNISHMENT = 0
WALL_BREAK_VALUE = None # make sure to change in the building blocks too


def find_empty_spaces(array):
    "takes an array and finds locations of 0's"
    return np.where(array == 0)


def find_all_possible_empty_locations(array):
    possible_goal_loc_V = find_empty_spaces(array)[0]
    possible_goal_loc_H = find_empty_spaces(array)[1]

    empty_locations = []
    for i in range(len(possible_goal_loc_V)):
        empty_locations.append([possible_goal_loc_V[i], possible_goal_loc_H[i]])

    return empty_locations


def find_agent_location(array):
    loc = np.where(array == 2)
    if len(loc[0]) > 1:
        print("debug point")
    try:
        return [loc[0][0], loc[1][0]]
    except:
        return [None, None]


def find_goal_location(array):
    loc = np.where(array == 3)
    try:
        return [loc[0][0], loc[1][0]]
    except:
        return None


def find_spikes_location(array):
    loc = np.where(array == -2)
    spike_locations = []
    try:
        for i in range(len(loc[0])):  # for multiple
            spike_locations.append([loc[0][i], loc[1][i]])
        return loc
    except:
        return None
    
def action_words_to_number(action_list, action):
    "action must arrive as a tuple for instance ('up',)"
    return action_list.index(action)


class Environment():
    def __init__(self, agent_location_value, goal_location_value, domain_size=[21, 21],
                 stochastic=False, stochastic_strength=None, max_episode_steps=4000,
                 style="grid", seed=0, random_start=False):
        
        "initiates all of the variables that we might need"
        
        
        self.initial_random_seed = True
        self.done = False
        # self.goal_location = [2, 2] # just init 
        self.step_counter = 0
        self.agent_location_value = agent_location_value
        self.goal_location_value = goal_location_value
        self.domain_size = domain_size
        self.stochastic = stochastic
        self.stochastic_strength = stochastic_strength
        self.style = style
        self.rand_seed = seed
        # self.rng = np.random.default_rng(seed=self.rand_seed)
        self.random_start = random_start
        # self.seed(seed)
        
        # UNHASH BELOW FOR RANDOM OR STRUCTURED DOMAIN BUILDING
        # ensure it makes sense with the starting location
        # self.random_domain_creation(domain_size)
        self.structured_domain_creation(domain_size)
        
        # Unsure if this is going to break something CHANGE ME
        self.original_domain = copy.deepcopy(self.domain)
        self.spike_locations = find_spikes_location(self.domain)  # this will be used later in the step.
        
        # UNHASH BELOW FOR RANDOM OR STRUCTURED STARTING LOCATIONS
        # ENSURE MAKES SENSE WITH THE STRUCTURED OR RANDOM DOMAIN
        if self.random_start:
            self.random_starting_location()
        else:
            self.structured_starting_location()
        
        
        self.original_starting_location_domain = copy.deepcopy(self.domain)
        
        # UNHASH BELOW FOR GOAL LOACTION STRATS
        # self.random_goal_location()
        # self.strategic_goal_location()
        self.random_strategic_goal_location(first=True)
        
        
        self.original_start_and_goal_location_domain = copy.deepcopy(self.domain)
        
        self.time_in_river = 0
        self.smash_counter = 0
        self.spikes = True
        self.spike_wait = 0
        self.previous_location_value = 0
        
        self.observation_space = self.get_observation_space()
        
        
    def seed(self, random_seed):
        # random.seed(random_seed)
        # np.random.seed(random_seed)
        # if random_seed:
        #     # print(random_seed)            
        
        self.rand_seed = random_seed
        
        self.structured_domain_creation(self.domain_size)
        
        self.original_domain = copy.deepcopy(self.domain)
        self.spike_locations = find_spikes_location(self.domain)  # this will be used later in the step.
        
        # UNHASH BELOW FOR RANDOM OR STRUCTURED STARTING LOCATIONS
        # ENSURE MAKES SENSE WITH THE STRUCTURED OR RANDOM DOMAIN
        
        if self.random_start:
            self.random_starting_location()
        else:
            self.structured_starting_location()
        
        
        self.original_starting_location_domain = copy.deepcopy(self.domain)
        
        # UNHASH BELOW FOR GOAL LOACTION STRATS
        # self.random_goal_location()
        # self.strategic_goal_location()
        self.random_strategic_goal_location(first=True)
        
        
        self.original_start_and_goal_location_domain = copy.deepcopy(self.domain)
        
        self.time_in_river = 0
        self.smash_counter = 0
        self.spikes = True
        self.spike_wait = 0
        self.previous_location_value = 0
        
        self.observation_space = self.get_observation_space()
        
    def get_dir(self, agent_loc, goal_loc):
        direction_temp = [None, None]
        direction_temp[0] = goal_loc[0] - agent_loc[0] # x
        direction_temp[1] = goal_loc[1] - agent_loc[1] # y
        
        if direction_temp[0] == 0:
            direction_temp[0] += 1e-5 # avoid division by zero
        
        # print(direction_temp)
        
        
        
        if (direction_temp[0] == 0) & (direction_temp[1] == 0): # on goal !!! CHECK THIS LATER
            direction = 0.05 # for debugging
        elif (direction_temp[0] >= 0) & (direction_temp[1] <= 0): # top right
            direction = np.arctan(abs(direction_temp[1])/abs(direction_temp[0]))
        elif (direction_temp[0] >= 0) & (direction_temp[1] >= 0): # bottom right
            direction = -np.arctan(abs(direction_temp[1])/abs(direction_temp[0]))
        elif (direction_temp[0] <= 0) & (direction_temp[1] <= 0): # top left
            direction = np.pi - np.arctan(abs(direction_temp[1])/abs(direction_temp[0]))
        elif (direction_temp[0] <= 0) & (direction_temp[1] >= 0): # bottom left
            direction = - np.pi + np.arctan(abs(direction_temp[1])/abs(direction_temp[0]))
        
        return direction
        
    def get_observation_space(self):
        
        # find the agent location
        # add a buffer of atleast 6 of walls around the edge
        # return an array which is the surroundings of the agent location
        
        obs_space = copy.deepcopy(self.domain)
        obs_space = np.pad(obs_space, ((3,3),(3,3)), "constant", constant_values=((1,1),(1,1)))
        obs_centre = find_agent_location(obs_space)
        agent_location = find_agent_location(self.domain)
        
        obs_space_1 = obs_space[obs_centre[0]-3:obs_centre[0]+4, obs_centre[1]-3:obs_centre[1]+4]
        
        goal_loc = find_goal_location(self.domain)
        
        direction_temp = [None, None]
        if goal_loc == None:
            plt.imshow(self.domain)
            debug = 1
        elif obs_centre == None:
            debug = 2
            
        direction = self.get_dir(agent_location, goal_loc)
        direction = direction / np.pi # scale this here
        
        magnitude = np.sqrt((goal_loc[0]-agent_location[0])**2 + (goal_loc[1]-agent_location[1])**2)
        magnitude = magnitude / self.domain_size[0] # scale here too
        
        # remove agent (as always in the centre)
        obs_space_1[3][3] = 0
        
        # remove goal location (as this is embedded in the dir and dis)
        if np.where(obs_space_1==3)[0].size != 0:
            goal_x = np.where(obs_space_1==3)[0][0]
            goal_y = np.where(obs_space_1==3)[1][0]
            obs_space_1[goal_x][goal_y] = 0
                
        # also remove the wall just because the VAE works better like this.
        if np.where(obs_space_1==4)[0].size != 0:
            goal_x = np.where(obs_space_1==4)[0][0]
            goal_y = np.where(obs_space_1==4)[1][0]
            obs_space_1[goal_x][goal_y] = 0
            
        obs_space_2 = torch.tensor([2*direction, 2*magnitude])
        
        obs_space = [obs_space_1, obs_space_2]
        
        return obs_space


    def strategic_goal_location(self):
        max_attempts = 100
        max_distance = 0
        max_goal_position = None
        for attempt in range(max_attempts):
            self.possible_choices = find_all_possible_empty_locations(self.domain)
            # choice_index = self.rng.choice(range(len(self.possible_choices)))
            choice_index = np.random.choice(range(len(self.possible_choices)))
            choice = self.possible_choices[choice_index]
            goal_position = [choice[0], choice[1]]
            distance = np.sqrt(
                (self.agent_location[0] - goal_position[0]) ** 2 + (self.agent_location[1] - goal_position[1]) ** 2)
            if distance > max_distance:
                max_distance = distance
                max_goal_position = goal_position
        
        # made change below to make the second arg just one braket
        self.domain[max_goal_position[0]][max_goal_position[1]] = self.goal_location_value
        self.goal_location = max_goal_position
        self.original_start_and_goal_location_domain = copy.deepcopy(self.domain)
        
    def random_strategic_goal_location(self, first = False):
        if not first:
            self.domain = copy.deepcopy(self.original_start_and_goal_location_domain)
        
        tgl = find_goal_location(self.domain)
        if tgl != None:
            self.goal_location = [tgl[0], tgl[1]]
            self.domain[self.goal_location[0]][[self.goal_location[1]]] = 0
            #print("found a goal already removing ...")
        
        self.agent_location = find_agent_location(self.domain)
        
        if self.random_start:
            self.random_starting_location()
        else:
            self.structured_starting_location()
        
        # print("Agents location should be 1,1 where are we?")
        # print(self.agent_location)
        
        threshold = 0.5 * self.domain_size[0]
        self.possible_choices = find_all_possible_empty_locations(self.domain)
        distance = 0
        while distance < threshold:
            # choice_index = self.rng.choice(range(len(self.possible_choices)))
            choice_index = np.random.choice(range(len(self.possible_choices)))
            choice = self.possible_choices[choice_index]
            goal_position = [choice[0], choice[1]]
            distance = np.sqrt(
                (self.agent_location[0] - goal_position[0]) ** 2 + (self.agent_location[1] - goal_position[1]) ** 2)
        
        self.domain[goal_position[0]][[goal_position[1]]] = self.goal_location_value
        self.goal_location = goal_position
        self.original_start_and_goal_location_domain = copy.deepcopy(self.domain)
        

    def random_domain_creation(self, domain_size):
        self.bbs = building_blocks_obstacles.get_building_blocks(style=self.style)
        self.possible_choices = []
        while self.possible_choices == []:  # needed because sometimes the domain creation will just fill all with 1's
            self.domain = domain_creation_obstacles.random_domain_creation(self.bbs, domain_size,
                                                                           style=self.style, seed=self.rand_seed)
            self.possible_choices = find_all_possible_empty_locations(self.domain)

    def structured_domain_creation(self, domain_size):
        self.bbs = building_blocks_obstacles.get_building_blocks(style=self.style)
        self.possible_choices = []
        while self.possible_choices == []:  # needed because sometimes the domain creation will just fill all with 1's
            self.domain = domain_creation_obstacles.structured_domain_creation(self.bbs, domain_size,
                                                                               style=self.style, seed=self.rand_seed)
            self.possible_choices = find_all_possible_empty_locations(self.domain)

    def random_starting_location(self):
        fal = find_agent_location(self.domain)
        if fal[0] is not None:
            self.domain[fal[0]][fal[1]] = 0
        
        self.possible_choices = find_all_possible_empty_locations(self.domain)
        # choice_index = self.rng.choice(range(len(self.possible_choices)))
        choice_index = np.random.choice(range(len(self.possible_choices)))
        choice = self.possible_choices[choice_index]
        starting_position = [choice[0], choice[1]]
        self.domain[starting_position[0]][starting_position[1]] = self.agent_location_value
        self.agent_location = starting_position

    def structured_starting_location(self):
        #check if an agent already exists in the environment and remove.
        fal = find_agent_location(self.domain)
        if fal[0] is not None:
            self.domain[fal[0]][fal[1]] = 0
                
        starting_position = [1, 1]
        self.domain[starting_position[0]][starting_position[1]] = self.agent_location_value
        self.agent_location = starting_position

    def random_goal_location(self):
        self.possible_choices = find_all_possible_empty_locations(self.domain)
        # choice_index = self.rng.choice(range(len(self.possible_choices)))
        choice_index = np.random.choice(range(len(self.possible_choices)))
        choice = self.possible_choices[choice_index]
        goal_position = [choice[0], choice[1]]
        self.domain[goal_position[0]][[goal_position[1]]] = self.goal_location_value
        self.goal_location = goal_position
        self.original_start_and_goal_location_domain = copy.deepcopy(self.domain)

    def reset(self, goal_choice=[], start_same=True, total_random=False, train=True, seed=None):
        """resets the environment goal_choice can be an empty list which means keep the same or a list of a curriculum
        of goal locations"""
        self.done = False
        self.previous_location_value = 0
        if not train:
            # r_seed = self.rng.randint(1001,2000)
            r_seed = np.random.randint(1001,2000)
            self.seed(r_seed)
            self.structured_domain_creation(self.domain_size)
            if self.random_start:
                self.random_starting_location()
            else:
                self.structured_starting_location()
            self.random_strategic_goal_location(first=True)
            
        elif start_same and (goal_choice == []):
            self.domain = copy.deepcopy(self.original_start_and_goal_location_domain)  # unchanged
            self.done = False
        elif start_same:  # this would be used if using a goal from a curriculum
            self.domain = copy.deepcopy(self.original_starting_location_domain)
            # goal_position_index = self.rng.choice(range(len(goal_choice)))
            goal_position_index = np.random.choice(range(len(goal_choice)))
            goal_position = goal_choice[goal_position_index]
            self.domain[goal_position[0]][goal_position[1]] = self.goal_location_value
            self.goal_location = goal_position
        elif total_random:
            self.domain = copy.deepcopy(self.original_domain)
            #self.random_starting_location()
            if self.random_start:
                self.random_starting_location()
            else:
                self.structured_starting_location()
            # self.random_goal_location()
            self.random_strategic_goal_location(first=True)
        elif seed:
            r_seed = seed
            self.seed(r_seed)
            # self.structured_domain_creation(self.domain_size)
            # self.structured_starting_location()
            # self.random_strategic_goal_location(first=True)
        else: # creates a brand new task
            if train:
                # r_seed = self.rng.randint(1, 100)
                r_seed = np.random.randint(1, 100)
            else:
                # r_seed = self.rng.randint(101,200)
                r_seed = np.random.randint(101,200)
            
            self.seed(r_seed)
            # self.structured_domain_creation(self.domain_size)
            # self.structured_starting_location()
            # self.random_strategic_goal_location(first=True)
            # self.domain = copy.deepcopy(self.original_domain)
            # self.random_starting_location()
            # # self.random_goal_location()
            # self.strategic_goal_location()

    def move_down(self):
        agent_loc = find_agent_location(self.domain)
        self.agent_location = [agent_loc[0], agent_loc[1]]
        new_agent_location = [agent_loc[0] + 1,
                              agent_loc[1]]  # +1 in the array indexing (higher index is actually down)
        if self.domain[new_agent_location[0]][
            new_agent_location[1]] == 0:  # check we can actually move into this location.
            self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
            self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
            self.agent_location = new_agent_location
            self.previous_location_value = 0
            return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == self.goal_location_value:  # check to see if we have reached our goal.
            self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
            self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
            self.agent_location = new_agent_location
            self.done = True
            return POSITIVE_REWARD
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == 1:  # check to see if we are moving into a wall, nothing happens if we are
            # need to check where we are though! 
            if self.domain[self.agent_location[0]][self.agent_location[1]] == -3:
                self.time_in_river += 1
                if self.time_in_river % 2 == 0:
                    ret_value = self.move_left()
                    self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                    return ret_value
            elif self.domain[self.agent_location[0]][self.agent_location[1]] == -5:
                self.time_in_river += 1
                if self.time_in_river % 2 == 0:
                    ret_value = self.move_down()
                    self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                    return ret_value
            return WALL_PUNISHMENT
        elif self.domain[new_agent_location[0]][new_agent_location[1]] == -1:  # we have died
            self.done = True
            return NEGATIVE_REWARD
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == -2:  # we have been spiked makes sure to change spikes before moving!
            self.done = True
            return NEGATIVE_REWARD
        elif self.domain[new_agent_location[0]][new_agent_location[
            1]] == -3:  # in the river flowing to the left, if in the river for more than 2 steps will
            self.time_in_river += 1
            if self.time_in_river % 2 == 0:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -3
                ret_value = self.move_left()
                self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                return ret_value
            else:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -3
                return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == -5:  # in the river flowing down, if in the river for more than 2 steps will
            self.time_in_river += 1
            if self.time_in_river % 2 == 0:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -5
                ret_value = self.move_down()
                self.time_in_river -= 1  # this is because move down will have added one aswell when it is unfair.
                return ret_value
            else:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -5
                return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == WALL_BREAK_VALUE:  # this is the one where we want to bang it down.
            self.smash_counter += 1
            if self.smash_counter > 1:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.smash_counter = 0
                self.previous_location_value = WALL_BREAK_VALUE
                return 0
            else:
                return 0  # do nothing

    def move_up(self):
        agent_loc = find_agent_location(self.domain)
        self.agent_location = [agent_loc[0], agent_loc[1]]
        new_agent_location = [agent_loc[0] - 1, agent_loc[1]]  # -1 in the array indexing
        if self.domain[new_agent_location[0]][
            new_agent_location[1]] == 0:  # check we can actually move into this location.
            self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
            self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
            self.agent_location = new_agent_location
            self.previous_location_value = 0
            return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == self.goal_location_value:  # check to see if we have reached our goal.
            self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
            self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
            self.agent_location = new_agent_location
            self.done = True
            return POSITIVE_REWARD
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == 1:  # check to see if we are moving into a wall, nothing happens if we are
            # need to check where we are though! 
            if self.domain[self.agent_location[0]][self.agent_location[1]] == -3:
                self.time_in_river += 1
                if self.time_in_river % 2 == 0:
                    ret_value = self.move_left()
                    self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                    return ret_value
            elif self.domain[self.agent_location[0]][self.agent_location[1]] == -5:
                self.time_in_river += 1
                if self.time_in_river % 2 == 0:
                    ret_value = self.move_down()
                    self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                    return ret_value
            return WALL_PUNISHMENT
        elif self.domain[new_agent_location[0]][new_agent_location[1]] == -1:  # we have died
            self.done = True
            return NEGATIVE_REWARD
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == -2:  # we have been spiked makes sure to change spikes before moving!
            self.done = True
            return NEGATIVE_REWARD
        elif self.domain[new_agent_location[0]][new_agent_location[
            1]] == -3:  # in the river flowing to the left, if in the river for more than 2 steps will
            self.time_in_river += 1
            if self.time_in_river % 2 == 0:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -3
                ret_value = self.move_left()
                self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                return ret_value
            else:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -3
                return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == -5:  # in the river flowing down, if in the river for more than 2 steps will
            self.time_in_river += 1
            if self.time_in_river % 2 == 0:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -5
                ret_value = self.move_down()
                self.time_in_river -= 1  # this is because move down will have added one aswell when it is unfair.
                return ret_value
            else:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -5
                return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == WALL_BREAK_VALUE:  # this is the one where we want to bang it down.
            self.smash_counter += 1
            if self.smash_counter > 1:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.smash_counter = 0
                self.previous_location_value = WALL_BREAK_VALUE
                return 0
            else:
                return 0  # do nothing

    def move_left(self):
        agent_loc = find_agent_location(self.domain)
        self.agent_location = [agent_loc[0], agent_loc[1]]
        new_agent_location = [agent_loc[0], agent_loc[1] - 1]  # -1 in the array indexing
        if self.domain[new_agent_location[0]][
            new_agent_location[1]] == 0:  # check we can actually move into this location.
            self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
            self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
            self.agent_location = new_agent_location
            self.previous_location_value = 0
            return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == self.goal_location_value:  # check to see if we have reached our goal.
            self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
            self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
            self.agent_location = new_agent_location
            self.done = True
            return POSITIVE_REWARD
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == 1:  # check to see if we are moving into a wall, nothing happens if we are
            # need to check where we are though! 
            if self.domain[self.agent_location[0]][self.agent_location[1]] == -3:
                self.time_in_river += 1
                if self.time_in_river % 2 == 0:
                    ret_value = self.move_left()
                    self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                    return ret_value
            elif self.domain[self.agent_location[0]][self.agent_location[1]] == -5:
                self.time_in_river += 1
                if self.time_in_river % 2 == 0:
                    ret_value = self.move_down()
                    self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                    return ret_value
            return WALL_PUNISHMENT
        elif self.domain[new_agent_location[0]][new_agent_location[1]] == -1:  # we have died
            self.done = True
            return NEGATIVE_REWARD
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == -2:  # we have been spiked makes sure to change spikes before moving!
            self.done = True
            return NEGATIVE_REWARD
        elif self.domain[new_agent_location[0]][new_agent_location[
            1]] == -3:  # in the river flowing to the left, if in the river for more than 2 steps will
            self.time_in_river += 1
            if self.time_in_river % 2 == 0:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -3
                ret_value = self.move_left()
                self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                return ret_value
            else:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -3
                return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == -5:  # in the river flowing down, if in the river for more than 2 steps will
            self.time_in_river += 1
            if self.time_in_river % 2 == 0:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -5
                ret_value = self.move_down()
                self.time_in_river -= 1  # this is because move down will have added one aswell when it is unfair.
                return ret_value
            else:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -5
                return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == WALL_BREAK_VALUE:  # this is the one where we want to bang it down.
            self.smash_counter += 1
            if self.smash_counter > 1:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.smash_counter = 0
                self.previous_location_value = WALL_BREAK_VALUE
                return 0
            else:
                return 0  # do nothing

    def move_right(self):
        agent_loc = find_agent_location(self.domain)
        self.agent_location = [agent_loc[0], agent_loc[1]]
        new_agent_location = [agent_loc[0], agent_loc[1] + 1]  # -1 in the array indexing
        if self.domain[new_agent_location[0]][
            new_agent_location[1]] == 0:  # check we can actually move into this location.
            self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
            self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
            self.agent_location = new_agent_location
            self.previous_location_value = 0
            return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == self.goal_location_value:  # check to see if we have reached our goal.
            self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
            self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
            self.agent_location = new_agent_location
            self.done = True
            return POSITIVE_REWARD
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == 1:  # check to see if we are moving into a wall, nothing happens if we are
            # need to check where we are though! 
            if self.domain[self.agent_location[0]][self.agent_location[1]] == -3:
                self.time_in_river += 1
                if self.time_in_river % 2 == 0:
                    ret_value = self.move_left()
                    self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                    return ret_value
            elif self.domain[self.agent_location[0]][self.agent_location[1]] == -5:
                self.time_in_river += 1
                if self.time_in_river % 2 == 0:
                    ret_value = self.move_down()
                    self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                    return ret_value
            return WALL_PUNISHMENT
        elif self.domain[new_agent_location[0]][new_agent_location[1]] == -1:  # we have died
            self.done = True
            return NEGATIVE_REWARD
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == -2:  # we have been spiked makes sure to change spikes before moving!
            self.done = True
            return NEGATIVE_REWARD
        elif self.domain[new_agent_location[0]][new_agent_location[
            1]] == -3:  # in the river flowing to the left, if in the river for more than 2 steps will
            self.time_in_river += 1
            if self.time_in_river % 2 == 0:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -3
                ret_value = self.move_left()
                self.time_in_river -= 1  # this is because move left will have added one aswell when it is unfair.
                return ret_value
            else:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -3
                return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == -5:  # in the river flowing down, if in the river for more than 2 steps will
            self.time_in_river += 1
            if self.time_in_river % 2 == 0:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -5
                ret_value = self.move_down()
                self.time_in_river -= 1  # this is because move down will have added one aswell when it is unfair.
                return ret_value
            else:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.previous_location_value = -5
                return 0
        elif self.domain[new_agent_location[0]][
            new_agent_location[1]] == WALL_BREAK_VALUE:  # this is the one where we want to bang it down.
            self.smash_counter += 1
            if self.smash_counter > 1:
                self.domain[new_agent_location[0]][new_agent_location[1]] = self.agent_location_value
                self.domain[self.agent_location[0]][self.agent_location[1]] = self.previous_location_value
                self.agent_location = new_agent_location
                self.smash_counter = 0
                self.previous_location_value = WALL_BREAK_VALUE
                return 0
            else:
                return 0  # do nothing

    def step(self, action):
        "this moves the agent as per the action decided by the agent, action must arrive as a list"
        # needs to return obs, reward, done, info
        # do I return a negative reward for choosing an action which hits a wall or do I just not allow it to happen?
        agent_loc = find_agent_location(self.domain)
        if agent_loc is None:
            print("can't find agent")
        self.agent_location = [agent_loc[0], agent_loc[1]]

        if self.spikes:
            if len(find_spikes_location(self.domain)[0]) != 0:
                for i in range(len(find_spikes_location(self.domain)[0])):
                    self.domain[self.spike_locations[0][i]][self.spike_locations[1][i]] = 0
                self.spike_wait = 0
                self.spikes = False
        elif self.spike_wait == 2:
            for i in range(len(self.spike_locations[0])):
                self.domain[self.spike_locations[0][i]][self.spike_locations[1][i]] = -2
                if self.domain[self.agent_location[0]][
                    self.agent_location[1]] == -2:  # agent has been caught out by the trap.
                    self.done = True
                    # print("agent has been killed by spikes")
                    return copy.deepcopy(self.domain), -1, self.done, []
            self.spikes = True
        else:
            self.spike_wait += 1

        reward_summer = []
        for a in action:  # actions arrive in a list because we will build the composite structure.
            if self.done:
                break
            # for debugging      
            # plt.imshow(self.domain)
            # plt.show()
            # for debugging end
            if self.stochastic:
                if self.rng.random() < self.stochastic_strength:
                    actions = ["up", "down", "left", "right"]
                    a = random.choice(actions)
            
            # print(a)
            # print(self.domain)
            
            if a == "up":
                t_r = self.move_up()
            elif a == "down":
                t_r = self.move_down()
            elif a == "left":
                t_r = self.move_left()
            elif a == "right":
                t_r = self.move_right()
            else:
                print("something has gone wrong; action is ", a)
            # reward_summer.append(-0.001)  # a small term to encourage faster solutions.
            
            if t_r is None:
                print("debug point")
            
            reward_summer.append(t_r)
            
            # For each primitive step a negative reward is given.            
            if self.done != True:
                reward_summer.append(STEP_REWARD)

        # print(reward_summer)
        return copy.deepcopy(self.domain), sum(
            reward_summer), self.done, []  
    
        # need to copy these because otherwise it just points


# below will be in separate files but for ease i'll put them in here for now.


if __name__ == "__main__":
    # The following needs to be copied into the console before you run each time
    # import sys
    # sys.path.append("C:/Users/x4nno/Documents/PhD/project_001/")
    env1 = Environment(2, 3, domain_size=[21, 21], style="Romesh_maze", seed=2)
    domain = env1.domain

    colors = ["white", "black", "blue", "yellow", "brown"]
    levels = [0, 1, 2, 3, 4, 5]
    
    cm, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
    
    plt.imshow(domain, cmap=cm, norm=norm, interpolation="none")
    plt.axis('off')
    plt.savefig("/home/x4nno/Documents/PhD/FRACOs_vg2/writing_images/romesh.svg", format='svg',
                bbox_inches='tight', dpi=300)
    plt.show()

    env1.step(("down",))

    plt.imshow(domain, cmap=cm, norm=norm, interpolation="none")
    plt.show()
