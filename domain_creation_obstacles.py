 # -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 13:57:14 2021

@author: x4nno
"""

path = "C:/Users/x4nno/Documents/PhD/project_001/"
import sys 

sys.path.append("/home/x4nno/Documents/PhD/MetaGridEnv/MetaGridEnv/")

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import random

from building_blocks_obstacles import get_building_blocks


bbs = get_building_blocks()

walls = np.array([[1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1]])

def random_domain_creation(bbs, size=[70,70], style="grid", seed=None, rng=None):
    "creates a random domain made from the building blocks - size is a vector which determins the shape of the maze."
    if rng is None:
        if seed:
            rng = np.random.randint(seed=seed)
        else:
            rng = np.random.randint()
    
    domain = 1*np.ones([size[0]+2,size[1]+2]) #initiation with a border of 1's 
    
    i_counter = 1 #horizontal
    j_counter = 8 #vertical
    random_index = np.random.choice(range(len(bbs)))
    starting_block = bbs[random_index]
    domain[1:8, 1:8] = starting_block  # starting from the top left corner. filling across and then down
    first = True
    while i_counter < size[0]+1:
        if first != True:
            j_counter = 1
        while j_counter < size[1]+1:
            if i_counter == 1:
                above_block_gaps = []  # no blocks above just a wall.
            else:
                above_block = domain[i_counter-7:i_counter, j_counter:j_counter+7]
                above_block_gaps = [index for index in range(7) if (above_block[6, index] == 0) or (above_block[6, index] == -2)]  # find where the gaps are in the block above
            if j_counter == 1:  # don't need to check to the left as we are up against the wall
                left_block_gaps = []  # no blocks to the left just a wall.
            else:
                left_block = domain[i_counter:i_counter+7, j_counter-7:j_counter]
                left_block_gaps = [index for index in range(7) if (left_block[index, 6] == 0) or (left_block[index, 6] == -2)]  # find where the gaps are in the block to left
            
            block_chosen = False
            loop_counter = 0
            while block_chosen != True:
                random_choice = rng.choice(range(len(bbs)))
                block_proposed = bbs[random_choice]
                block_proposed_top_gaps = [index for index in range(7) if (block_proposed[0, index] == 0) or (block_proposed[0, index] == -2)]  # check gaps at the top of this block
                block_proposed_left_gaps = [index for index in range(7) if (block_proposed[index, 0] == 0) or (block_proposed[index, 0] == -2)]  # check gaps at the left of this block
                
                for compare1 in range(7):  # compare if it can connect to the left
                    if (compare1 in block_proposed_left_gaps) and (compare1 in left_block_gaps):
                        block_chosen = True
                for compare2 in range(7):  # compare if it can connect above (we only need on or the other to be true)
                    if (compare2 in block_proposed_top_gaps) and (compare2 in above_block_gaps):
                        block_chosen = True
                
                if i_counter == 1:  # can't have a block on the right side otherwise it can't fill correctly.
                    block_proposed_right_gaps = [index for index in range(7) if (block_proposed[index, 2] == 0) or (block_proposed[index, 2] == -2) ]
                    if block_proposed_right_gaps == []:
                        block_chosen = False
                if j_counter == 1:  # can't have a block on the bottom else it won't fill correctly.
                    block_proposed_bottom_gaps = [index for index in range(7) if (block_proposed[2, index] == 0) or (block_proposed[2, index] == -2)]
                    if block_proposed_bottom_gaps == []:
                        block_chosen = False
                if loop_counter > 3*len(bbs):
                    block_proposed = walls
                    block_chosen = True
                loop_counter += 1
            domain[i_counter:i_counter+7, j_counter:j_counter+7] = block_proposed
            j_counter += 7
            first = False
            # print(domain)
        i_counter += 7
    
    domain_temp = np.ravel(domain)

    return domain


def structured_domain_creation(bbs, size=[70, 70], reward_block_indexs=[3,6,7],
                               style="grid", seed=None, rng=None):
    
    if style in ["Four_rooms", "Josh_grid", "Romesh_maze"]:
        return bbs[0]
    
    if rng is None:
        if seed:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()
    
    domain = 1 * np.ones([size[0] + 2, size[1] + 2])  # initiation with a border of 1's

    i_counter = 1  # horizontal
    j_counter = 8  # vertical
    reward_block_index = rng.choice(reward_block_indexs)
    starting_block = bbs[reward_block_index]
    domain[1:8, 1:8] = starting_block  # starting from the top left corner. filling across and then down
    first = True
    while i_counter < size[0] + 1:
        if first != True:
            j_counter = 1
        while j_counter < size[1] + 1:
            if i_counter == 1:
                above_block_gaps = []  # no blocks above just a wall.
            else:
                above_block = domain[i_counter - 7:i_counter, j_counter:j_counter + 7]
                above_block_gaps = [index for index in range(7) if (above_block[6, index] == 0) or (
                            above_block[6, index] == -2)]  # find where the gaps are in the block above
            if j_counter == 1:  # don't need to check to the left as we are up against the wall
                left_block_gaps = []  # no blocks to the left just a wall.
            else:
                left_block = domain[i_counter:i_counter + 7, j_counter - 7:j_counter]
                left_block_gaps = [index for index in range(7) if (left_block[index, 6] == 0) or (
                            left_block[index, 6] == -2)]  # find where the gaps are in the block to left

            block_chosen = False
            loop_counter = 0
            while block_chosen != True:
                random_choice = rng.choice(range(len(bbs)))
                block_proposed = bbs[random_choice]
                block_proposed_top_gaps = [index for index in range(7) if (block_proposed[0, index] == 0) or (
                            block_proposed[0, index] == -2)]  # check gaps at the top of this block
                block_proposed_left_gaps = [index for index in range(7) if (block_proposed[index, 0] == 0) or (
                            block_proposed[index, 0] == -2)]  # check gaps at the left of this block

                for compare1 in range(7):  # compare if it can connect to the left
                    if (compare1 in block_proposed_left_gaps) and (compare1 in left_block_gaps):
                        block_chosen = True
                for compare2 in range(7):  # compare if it can connect above (we only need on or the other to be true)
                    if (compare2 in block_proposed_top_gaps) and (compare2 in above_block_gaps):
                        block_chosen = True

                if i_counter == 1:  # can't have a block on the right side otherwise it can't fill correctly.
                    block_proposed_right_gaps = [index for index in range(7) if
                                                 (block_proposed[index, 2] == 0) or (block_proposed[index, 2] == -2)]
                    if block_proposed_right_gaps == []:
                        block_chosen = False
                if j_counter == 1:  # can't have a block on the bottom else it won't fill correctly.
                    block_proposed_bottom_gaps = [index for index in range(7) if
                                                  (block_proposed[2, index] == 0) or (block_proposed[2, index] == -2)]
                    if block_proposed_bottom_gaps == []:
                        block_chosen = False
                if loop_counter > 3 * len(bbs):
                    block_proposed = walls
                    block_chosen = True
                loop_counter += 1
            domain[i_counter:i_counter + 7, j_counter:j_counter + 7] = block_proposed
            j_counter += 7
            first = False
            # print(domain)
        i_counter += 7

    domain_temp = np.ravel(domain)

    return domain
                

if __name__ == "__main__":
    bbs = get_building_blocks(style="Romesh_maze")
    
    size = [14, 14]
    domain1 = structured_domain_creation(bbs, size = size, style="Romesh_maze", seed=42)
    
    bbs = get_building_blocks(style="Romesh_maze")
    domain2 = structured_domain_creation(bbs, size = size, style="Romesh_maze", seed=43)
    
    colors = ["brown", "white", "black", "blue", "yellow"]
    levels = [-4, 0, 1, 2, 3, 4]
    
    cm, norm = matplotlib.colors.from_levels_and_colors(levels, colors)

    plt.imshow(domain1, cmap = cm, norm=norm, interpolation="none")
    plt.show()
    
    plt.imshow(domain2, cmap = cm, norm=norm, interpolation="none")
    plt.show()
                