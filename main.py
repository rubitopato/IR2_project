from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from perceptual_space import get_perceptual_state, get_perceptual_state_limited
from action_space import perform_random_action_freely, perform_random_action_limited
from utils import get_cylinders_initial_pos, move_cylinder, reset_position_cylinders
import time
import random

## Initialization of the robot
rob = Robobo("localhost")
sim = RoboboSim("localhost")
sim.connect()
rob.connect()

rob.playNote(59, 0.08, wait=True)
time.sleep(0.05)
rob.playNote(64, 0.08, wait=True)
time.sleep(0.05)
rob.playNote(69, 0.08, wait=True)

# time.sleep(0.7)
# rob.playNote(69, 0.18, wait=True)
# time.sleep(0.12)
# rob.playNote(69, 0.18, wait=True)

rob.setEmotionTo(Emotions.ANGRY)  #DO NOT TOUCH THIS LINE
rob.moveTiltTo(100,50)

## Testing the movement of the cylinders
objects = sim.getObjects()
cylinders_initial_pos = get_cylinders_initial_pos(sim, objects)

for i in range(10):
    cylinder_name = random.choice(list(cylinders_initial_pos.keys()))
    print("Moving cylinder:", cylinder_name)
    move_cylinder(sim, cylinder_name)
    time.sleep(1)

reset_position_cylinders(sim, cylinders_initial_pos)

## Execution of the robot
# get_perceptual_state_limited(sim)

# i = 0
# while i < 100:
#     get_perceptual_state_limited(sim)
#     r, l = perform_random_action_freely(rob)
#     print(r, l)
#     i += 1
 
rob.disconnect()
sim.disconnect()