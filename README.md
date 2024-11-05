# MetaGridEnv


THE IMAGES HAVEN@T TRANSFERRED: VISIT HERE FOR A BETTER README: https://threebody.notion.site/MetaGridEnv-f970ea9028cf4db88f6fd252be1b454b?pvs=4

This environment creates distinctly different grid-worlds, but each are built with shared building blocks. Therefore this provides structure for an agent to learn, whilst also demonstrating it’s ability to extrapolate to unseen domains.

### INSTALLATION and USAGE

1. Create and activate a new virtual environment 

```bash
conda create -n <env_name> 
conda activate <env_name>
```

1. Clone (download) this repo and unzip 
2. Navigate (in terminal) to the unzipped MetaGridEnv environment
3. Install in the terminal using the following: 
    
    (If you have any requirement issues then install those too)
    

```bash

pip install .
```

1. Now add the following block to the top of your script: 

```python
import sys
# The path below needs to point to the folder which is MetaGridEnv/MetaGridEnv
sys.path.append("<path to where MetaGridEnv/MetaGridEnv is>")
import MetaGridEnv
from gym.envs.registration import register 

#  Register only needs to be run once (but everytime the script is run)

register( id="MetaGridEnv/metagrid-v0",
          entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")
```

1. Create your gym environment as below, note the following:
    1. style= “grid” or “Tori” (explanation of difference is below)
    2. domain_size must be a multiple of 7. If choosing style “Tori” then the first digit in domain_size must be 7.

```python
env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="grid")
```

1. Use as you would use any Gym Environment. Reward is 5 for reaching the goal and -0.1 per timestep. 

### Style Types:

1. Grid: running style=”grid” will give you domains as below

```python
env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[21,21], style="grid")
```

The top demonstrates the building blocks that each are made from, and the bottom demonstrates possible domain combinations of these building blocks.
![Untitled](images/grid.png)



1. Tori: running style=”Tori” will give you domains as below:
    
    This is useful for doing things such as diverse density prediction of subgoals.
    

```python
env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[7, 21], style="Tori")
```

![Untitled](images/tori.png)
