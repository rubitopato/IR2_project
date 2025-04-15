from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor
from perceptual_space import get_perceptual_state, get_perceptual_state_limited
from action_space import perform_random_action_freely, perform_random_action_limited
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
 
rob = Robobo("localhost")
sim = RoboboSim("localhost")
sim.connect()
rob.connect()
rob.setEmotionTo(Emotions.ANGRY) # DO NOT TOUCH THIS LINE
rob.moveTiltTo(100,50)

i = 0
while i < 100:
    get_perceptual_state_limited(sim)
    r, l = perform_random_action_freely(rob)
    print(r, l)
    i += 1
 
rob.disconnect()
sim.disconnect()