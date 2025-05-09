from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from perceptual_space import get_perceptual_state_limited
from action_space import generate_all_possible_actions
from utils import avoid_obstacle, objective_found, get_array_from_perceptual_dict, get_intrinsinc_utility_from_state
from utils import predict_multiple_next_states_batched, load_scaler, load_world_model
from robobopy.utils.IR import IR
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

MIN_NUM_TRACES = 20
fit_scaler = True

memory_of_states_visited = []
traces = []
traces_utilities = []
current_trace = []
current_trace_utilities = []
possible_actions = generate_all_possible_actions(-30, 30)
world_model = load_world_model("src/models/xgb_world_model_v7.joblib")
scaler = load_scaler("src/models/xgb_scaler_v7.joblib")
extrinsic_utility_model = joblib.load('src/models/extrinsic_utility_model3.pkl')

# while len(traces) < MIN_NUM_TRACES:
    
#     if rob.readIRSensor(IR.FrontC) > 100 or rob.readIRSensor(IR.BackC) > 110:
#         print("avoiding obstacle")
#         avoid_obstacle(rob, rob.readIRSensor(IR.FrontC), rob.readIRSensor(IR.BackC))
#         rob.wait(1)
#     else:
#         perception_init = get_perceptual_state_limited(sim)
#         print("Initial Red Distance: ", perception_init.get('distance_red'))
#         print("Initial Red Angle: ", perception_init.get('angle_red'))
#         predicted_states, actions = predict_multiple_next_states_batched(perception_init, possible_actions, world_model, scaler)
        
#         found_goal, index_best_action = objective_found(predicted_states)
        
#         if found_goal: 
#             best_action = actions[index_best_action]   
#             rob.moveWheelsByTime(best_action[0], best_action[1], 1, wait=True) 
#             current_state_list = get_array_from_perceptual_dict(get_perceptual_state_limited(sim))
#             memory_of_states_visited.append(current_state_list)
#             current_trace.append(current_state_list)
            
#             current_trace = current_trace[-10:]
#             length = len(current_trace)
#             for i, state in enumerate(current_trace):
#                 utility = (i + 1) / length  # 0 en el inicio, 1 en el objetivo
#                 current_trace_utilities.append(utility)   
                
#             if len(current_trace) < 10:
#                 padding = [[1000,200,1000,200,1000,200]] * (10 - len(current_trace))
#                 current_trace = padding + current_trace 
#                 padding = [0.001] * (10 - len(current_trace)) 
#                 current_trace_utilities = padding + current_trace_utilities
                
#             traces_utilities.append(current_trace_utilities.copy())
#             traces.append(current_trace.copy())      
            
#             line = f'traces: {current_trace}\n'
#             line += f'traces utilities: {current_trace_utilities}'
#             with open("dataset/new_traces.txt", "a") as archivo:
#                 archivo.write(f"\n{line}")
#             current_trace_utilities.clear()
#             current_trace.clear()
#             memory_of_states_visited.clear()

#             sim.resetSimulation()      
            
#             rob.wait(1) 
            
#             continue
            
#         else:
#             intrinsic_utilities = [get_intrinsinc_utility_from_state(predicted_state, memory_of_states_visited, 1) for predicted_state in predicted_states]
#             best_action = actions[intrinsic_utilities.index(max(intrinsic_utilities))]
#             print("Predicted Red Distance: ", predicted_states[np.argmax(intrinsic_utilities)].get('distance_red'))
#             print("Predicted Red Angle: ", predicted_states[np.argmax(intrinsic_utilities)].get('angle_red'))
#             print("Predicted utility: ", max(intrinsic_utilities))
        
#             print("Best action: ", best_action)
#             rob.moveWheelsByTime(best_action[0], best_action[1], 1, wait=True)
#             perception_after_action = get_perceptual_state_limited(sim)
#             print("Final Red Distance: ", perception_after_action.get('distance_red'))
#             print("Final Red Angle: ", perception_after_action.get('angle_red'))
#             print("Real utility: ", get_intrinsinc_utility_from_state(perception_after_action, memory_of_states_visited, 1))
#             current_state_list = get_array_from_perceptual_dict(perception_after_action)
#             memory_of_states_visited.append(current_state_list)
#             current_trace.append(current_state_list)
#             print("--------------------")
                    
            
#         rob.wait(0.1)
        
#     if len(memory_of_states_visited) >= 5:
#         memory_of_states_visited = memory_of_states_visited[-5:]

# extrinsic_utility_model = train_model(extrinsic_utility_model, traces, traces_utilities, True)
# joblib.dump(extrinsic_utility_model, 'src/models/extrinsic_utility_model3.pkl')

round_n = 0
while round_n < 5:
    if rob.readIRSensor(IR.FrontC) > 100 or rob.readIRSensor(IR.BackC) > 110:
        print("avoiding obstacle")
        avoid_obstacle(rob, rob.readIRSensor(IR.FrontC), rob.readIRSensor(IR.BackC))
        rob.wait(1)
    else:
        perception_init = get_perceptual_state_limited(sim)
        print("Initial Red Distance: ", perception_init.get('distance_red'))
        print("Initial Red Angle: ", perception_init.get('angle_red'))
        predicted_states, actions = predict_multiple_next_states_batched(perception_init, possible_actions, world_model, scaler)
        
        found_goal, index_best_action = objective_found(predicted_states)
        
        if found_goal:
            print(index_best_action)
            best_action = actions[index_best_action]   
            rob.moveWheelsByTime(best_action[0], best_action[1], 1, wait=True)
            print("Objetive reached")
            round_n += 1
            sim.wait(1)
            rob.wait(1)
            sim.resetSimulation()      
            
            rob.wait(1) 
            rob.wait(1)
            continue
        else:     
            extrinsic_utilities = np.array(get_extrinsic_utility_from_states(predicted_states, extrinsic_utility_model))          
            best_action = actions[np.argmax(extrinsic_utilities)]
            print("Predicted Red Distance: ", predicted_states[np.argmax(extrinsic_utilities)].get('distance_red'))
            print("Predicted Red Angle: ", predicted_states[np.argmax(extrinsic_utilities)].get('angle_red'))
            print("Predicted extrinsic utility: ",extrinsic_utilities.max())
            rob.moveWheelsByTime(best_action[0], best_action[1], 1, wait=True) 
            perception_final = get_perceptual_state_limited(sim)
            found_goal, index_best_action = objective_found([perception_final])
            print("Final Red Distance: ", perception_final.get('distance_red'))
            print("Final Red Angle: ", perception_final.get('angle_red'))
            if found_goal:
                print(index_best_action)
                best_action = actions[index_best_action]   
                rob.moveWheelsByTime(best_action[0], best_action[1], 1, wait=True)
                print("Objetive reached")
                round_n += 1
                sim.wait(1)
                rob.wait(1)
                sim.resetSimulation()      
                
                rob.wait(1) 
                rob.wait(1)
                continue
            
            print("-------------")
 
rob.disconnect()
sim.disconnect()