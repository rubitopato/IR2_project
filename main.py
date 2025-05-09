from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from perceptual_space import get_perceptual_state, get_perceptual_state_limited
from action_space import perform_random_action_freely, perform_random_action_limited
from utils import get_cylinders_initial_pos, move_cylinder, reset_position_cylinders, save_new_line_of_data, avoid_obstacle, reset_randomize_positions
import time
from robobopy.utils.IR import IR
from test_word_model import test_world_model

## Initialization of the robot
rob = Robobo("localhost")
sim = RoboboSim("localhost")
sim.connect()
rob.connect()

rob.setEmotionTo(Emotions.ANGRY)  #DO NOT TOUCH THIS LINE
rob.moveTiltTo(100,50)

#test_world_model(rob, sim)

i = 0
j = 0
while i < 60:
    
    while j < 50:
        if rob.readIRSensor(IR.FrontC) > 60 or rob.readIRSensor(IR.BackC) > 85:
            print("avoiding obstacle")
            avoid_obstacle(rob, rob.readIRSensor(IR.FrontC), rob.readIRSensor(IR.BackC))
            rob.wait(1)
        else:
            perception_init = get_perceptual_state_limited(sim)
            r, l = perform_random_action_freely(rob)
            rob.stopMotors()
            perception_final = get_perceptual_state_limited(sim)
            save_new_line_of_data(perception_init, r, l, perception_final)
            j += 1
            rob.wait(0.1)
            
    objects = sim.getObjects()
    reset_randomize_positions(sim, objects, 250)
    rob.wait(1)    
    i += 1
    j = 0
 
rob.disconnect()
sim.disconnect()