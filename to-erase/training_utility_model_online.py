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

MIN_NUM_TRACES = 15
fit_scaler = True

memory_of_states_visited = []
traces = []
traces_utilities = []
current_trace = []
current_trace_utilities = []
possible_actions = [(-30,-30),(0,-30),(-30,0),(0,30),(30,0),(30,30)]  #generate_all_possible_actions(-30, 30)
world_model = load_world_model("src/models/world_model_v3.joblib")
scaler = load_scaler("src/models/scaler_v3.joblib")
extrinsic_utility_model = get_extrinsic_utility_model()

while len(traces) < MIN_NUM_TRACES:
    
    if rob.readIRSensor(IR.FrontC) > 100 or rob.readIRSensor(IR.BackC) > 110:
        print("avoiding obstacle")
        avoid_obstacle(rob, rob.readIRSensor(IR.FrontC), rob.readIRSensor(IR.BackC))
        rob.wait(1)
    else:
        perception_init = get_perceptual_state_limited(sim)
        predicted_states, actions = predict_multiple_next_states_batched(perception_init, possible_actions, world_model, scaler)
        
        found_goal, index_best_action = objective_found(predicted_states)
        
        if found_goal: 
            print(index_best_action)
            best_action = actions[index_best_action]   
            rob.moveWheelsByTime(best_action[0], best_action[1], 1, wait=True) 
            current_state_list = get_array_from_perceptual_dict(get_perceptual_state_limited(sim))
            memory_of_states_visited.append(current_state_list)
            current_trace.append(current_state_list)
            
            current_trace = current_trace[-10:]
            length = len(current_trace)
            for i, state in enumerate(current_trace):
                utility = (i + 1) / length  # 0 en el inicio, 1 en el objetivo
                current_trace_utilities.append(utility)     
            traces_utilities.append(current_trace_utilities.copy())
            traces.append(current_trace.copy())      
            extrinsic_utility_model = train_model(extrinsic_utility_model, current_trace, current_trace_utilities, fit_scaler)
            fit_scaler = False
            line = f'traces: {current_trace}\n'
            line += f'traces utilities: {current_trace_utilities}'
            with open("dataset/new_traces.txt", "a") as archivo:
                archivo.write(f"\n{line}")
            current_trace_utilities.clear()
            current_trace.clear()
            memory_of_states_visited.clear()

            sim.resetSimulation()      
            
            rob.wait(1) 
            
            continue
            
        else:
            intrinsic_utilities = [get_intrinsinc_utility_from_state(predicted_state, memory_of_states_visited, 1.5) for predicted_state in predicted_states]
        
            if len(traces) > 0:
                extrinsic_utilities = get_extrinsic_utility_from_states(predicted_states, extrinsic_utility_model)    
                alpha = min(1.0, len(traces) / MIN_NUM_TRACES)
                print(alpha)
                final_utilities = alpha * np.array(extrinsic_utilities) + (1 - alpha) * np.array(intrinsic_utilities)           
                best_action = actions[np.argmax(final_utilities)]
                print("Predicted extrinsic utility: ",extrinsic_utilities[np.argmax(final_utilities)])
                print("Highest extrinsic utility: ",max(extrinsic_utilities))
                print("Best utility: ", max(final_utilities))
            else:
                best_action = actions[intrinsic_utilities.index(max(intrinsic_utilities))]
                print("Best utility: ", max(intrinsic_utilities))
        
            print("Best action: ", best_action)
            rob.moveWheelsByTime(best_action[0], best_action[1], 1, wait=True)
            perception_after_action = get_perceptual_state_limited(sim)
            print("Final Red Distance: ", perception_after_action.get('distance_red'))
            print("Final Red Angle: ", perception_after_action.get('angle_red'))
            print("Real utility: ", get_intrinsinc_utility_from_state(perception_after_action, memory_of_states_visited, 1.5))
            current_state_list = get_array_from_perceptual_dict(perception_after_action)
            memory_of_states_visited.append(current_state_list)
            current_trace.append(current_state_list)
            print("--------------------")
                    
            
        rob.wait(0.1)
        
    if len(memory_of_states_visited) >= 20:
        memory_of_states_visited = memory_of_states_visited[-20:]
        
joblib.dump(extrinsic_utility_model, 'src/models/extrinsic_utility_model2.pkl')
 
rob.disconnect()
sim.disconnect()