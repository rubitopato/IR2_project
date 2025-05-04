from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from perceptual_space import get_perceptual_state, get_perceptual_state_limited
from action_space import perform_random_action_freely, perform_random_action_limited
from utils import get_cylinders_initial_pos, move_cylinder, reset_position_cylinders, save_new_line_of_data, avoid_obstacle, objective_found, get_array_from_perceptual_dict, get_intrinsinc_utility_from_state
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

MIN_NUM_TRACES = 10

memory_of_states_visited = []
traces = []
traces_utilities = []
current_trace = []
current_trace_utilities = []

while True:
    
    if rob.readIRSensor(IR.FrontC) > 60 or rob.readIRSensor(IR.BackC) > 85:
        print("avoiding obstacle")
        avoid_obstacle(rob, rob.readIRSensor(IR.FrontC), rob.readIRSensor(IR.BackC))
        rob.wait(1)
    else:
        perception_init = get_perceptual_state_limited(sim)
        # predicted_states, actions = get_predicted_states(perception_init)
        # utilities = [get_utility_from_state(predicted_state) for predicted_state in predicted_states]
        
        # utilities = []
        # for predicted_state in predicted_states:
        #     intrinsic_utility = get_intrinsic_utility_from_state(predicted_state, memory_of_states_visited)
        #     extrinsic_utility = get_extrinsic_utility_from_state(predicted_state)
        #     alpha = min(1.0, len(traces) / MIN_NUM_TRACES)
        #     final_utility = alpha * extrinsic_utility + (1 - alpha) * intrinsic_utility
        #     utilities.append(final_utility)
        # best_action = actions[utilities.index(max(utilities))]
        # rob.moveWheelsByTime(best_action[0], best_action[1], 1, wait=True)
        perception_after_action = get_perceptual_state_limited(sim)
        memory_of_states_visited.append(perception_after_action)
        current_state_list = get_array_from_perceptual_dict(perception_after_action)
        current_trace.append(current_state_list)
        
        if objective_found(perception_after_action):     
            length = len(current_trace)
            for i, state in enumerate(current_trace):
                utility = (i + 1) / length  # 0 en el inicio, 1 en el objetivo
                current_trace_utilities.append(utility)     
            traces_utilities.append(current_trace_utilities.copy())
            traces.append(current_trace.copy())  
            current_trace_utilities.clear()
            current_trace.clear()
            
            # entrenar modelo con traces y traces_utilities
            
            # simulation reset       
            
            rob.wait(1)             
            
        rob.wait(0.1)
 
rob.disconnect()
sim.disconnect()