from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from perceptual_space import get_perceptual_state, get_perceptual_state_limited
from action_space import perform_random_action_freely, perform_random_action_limited, generate_all_possible_actions
from utils import get_cylinders_initial_pos, move_cylinder, reset_position_cylinders, save_new_line_of_data, avoid_obstacle, objective_found, get_array_from_perceptual_dict, get_intrinsinc_utility_from_state
from utils import predict_multiple_next_states_batched, load_scaler, load_world_model, reset_randomize_positions
import time
from robobopy.utils.IR import IR
from test_word_model import test_world_model
from utility_model import get_extrinsic_utility_model, get_extrinsic_utility_from_states, train_model
import joblib
import numpy as np

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
possible_actions = generate_all_possible_actions(-30, 30)
world_model = load_world_model("src/models/world_model_v3.joblib")
scaler = load_scaler("src/models/scaler_v3.joblib")
extrinsic_utility_model = get_extrinsic_utility_model()

while len(traces) < 10:
    
    if rob.readIRSensor(IR.FrontC) > 60 or rob.readIRSensor(IR.BackC) > 80:
        print("avoiding obstacle")
        avoid_obstacle(rob, rob.readIRSensor(IR.FrontC), rob.readIRSensor(IR.BackC))
        rob.wait(1)
    else:
        perception_init = get_perceptual_state_limited(sim)
        predicted_states, actions = predict_multiple_next_states_batched(perception_init, possible_actions, world_model, scaler)
        intrinsic_utilities = [get_intrinsinc_utility_from_state(predicted_state, memory_of_states_visited, 1.5) for predicted_state in predicted_states]
        if len(traces) > 0:
            extrinsic_utilities = get_extrinsic_utility_from_states(predicted_states, extrinsic_utility_model)    
            alpha = min(1.0, len(traces) / MIN_NUM_TRACES)
            print(alpha)
            final_utilities = alpha * np.array(extrinsic_utilities) + (1 - alpha) * np.array(intrinsic_utilities)           
            best_action = actions[np.argmax(final_utilities)]
            print("Best utility: ", max(final_utilities))
        else:
            best_action = actions[intrinsic_utilities.index(max(intrinsic_utilities))]
            print("Best utility: ", max(intrinsic_utilities))
        print("Distance: ", perception_init.get('distance_red'))
        
        print("Best action: ", best_action)
        rob.moveWheelsByTime(best_action[0], best_action[1], 1, wait=False)
        perception_after_action = get_perceptual_state_limited(sim)
        print("Real utility: ", get_intrinsinc_utility_from_state(perception_after_action, memory_of_states_visited, 1.5))
        current_state_list = get_array_from_perceptual_dict(perception_after_action)
        memory_of_states_visited.append(current_state_list)
        current_trace.append(current_state_list)
        print("--------------------")
        if objective_found(perception_after_action):     
            length = len(current_trace)
            for i, state in enumerate(current_trace):
                utility = (i + 1) / length  # 0 en el inicio, 1 en el objetivo
                current_trace_utilities.append(utility)     
            traces_utilities.append(current_trace_utilities.copy())
            traces.append(current_trace.copy())      
            print(traces)
            print(traces_utilities)
            extrinsic_utility_model = train_model(extrinsic_utility_model, current_trace, current_trace_utilities)
            
            current_trace_utilities.clear()
            current_trace.clear()

            reset_randomize_positions(sim, sim.getObjects())       
            
            rob.wait(1)             
            
        rob.wait(0.1)
        
        if len(memory_of_states_visited) >= 10:
            memory_of_states_visited = memory_of_states_visited[-10:]
        
extrinsic_utility_model.save("src/models/extrinsic_utility_model.h5")
 
rob.disconnect()
sim.disconnect()