from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor
from perceptual_space import get_perceptual_state, get_perceptual_state_limited
from action_space import perform_random_action_freely, perform_random_action_limited
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from robobopy.utils.IR import IR
from utils import avoid_obstacle
 
rob = Robobo("localhost")
sim = RoboboSim("localhost")
sim.connect()
sim.wait(1)
rob.connect()
rob.wait(1)
rob.setEmotionTo(Emotions.ANGRY) # DO NOT TOUCH THIS LINE
rob.moveTiltTo(100,50)

i = 0
while i < 100:
    while rob.readIRSensor(IR.FrontC) < 60:
        get_perceptual_state_limited(sim)
        r, l = perform_random_action_freely(rob)
        print(r, l)
        i += 1
        rob.wait(1)
        
    else:
        avoid_obstacle(rob)
        rob.wait(1)
 
rob.disconnect()
sim.disconnect()