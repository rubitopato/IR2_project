from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils.utils import get_array_from_perceptual_dict, avoid_obstacle, objective_found, get_intrinsinc_utility_from_state
from utils.simulation_utils import reset_randomize_positions
from src.perceptual_space import get_perceptual_state_limited
from src.world_model import predict_multiple_next_states_batched
from src.action_space import generate_all_possible_actions
import joblib
from robobopy.utils.IR import IR

# INTRINSIC UTILITY MODEL

def novelty_score(predicted_state, memory, n=1):

    if not memory:
        return 1000.0 

    try:
        pred_vector = np.array([
            predicted_state.get('distance_red', 0), predicted_state.get('angle_red', 0),
            predicted_state.get('distance_green', 0), predicted_state.get('angle_green', 0),
            predicted_state.get('distance_blue', 0), predicted_state.get('angle_blue', 0)
        ])

        total_distance = 0.0
        num_valid_past = 0
        for past_state in memory:
            if isinstance(past_state, dict):
                past_vector = np.array([
                    past_state.get('distance_red', 0), past_state.get('angle_red', 0),
                    past_state.get('distance_green', 0), past_state.get('angle_green', 0),
                    past_state.get('distance_blue', 0), past_state.get('angle_blue', 0)
                ])
                dist = np.linalg.norm(pred_vector - past_vector)
                total_distance += dist ** n
                num_valid_past += 1
            else:
                print("Warning: Item in memory is not a dictionary, skipping.")

        if num_valid_past == 0:
             return 1000.0

        novelty = total_distance / num_valid_past
        return novelty

    except Exception as e:
        print(f"Error during novelty calculation: {e}")
        return 0.0
    
# EXTRINSIC UTILITY MODEL

def get_extrinsic_utility_model():

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        learning_rate="adaptive",
    )
    return model

def train_model(x, y, utility_scaler_path):
    new_scaler = StandardScaler()
    model = get_extrinsic_utility_model()
    x = np.vstack(x)
    y = np.array([item for sublist in y for item in sublist])
    new_scaler.fit(x)
    joblib.dump(new_scaler, utility_scaler_path)
        
    x_scaled = new_scaler.transform(x)
    model.fit(x_scaled, y)
    return model, new_scaler

def partial_train_model(model, x, y, scaler_learned):
    x = np.vstack(x)
    y = np.array([item for sublist in y for item in sublist])
        
    x_scaled = scaler_learned.transform(x)
    model.partial_fit(x_scaled, y)
    return model

def get_extrinsic_utility_from_states(predicted_states, model, scaler_learned):
    predicted_states_array = [get_array_from_perceptual_dict(predicted_state) for predicted_state in predicted_states]
    predicted_states_numpy_array = np.array(predicted_states_array)
    predicted_states_scaled = scaler_learned.transform(predicted_states_numpy_array)
    utilities = model.predict(predicted_states_scaled)
    return utilities


def train_extrinsic_utility_model(sim, rob, min_num_traces, world_model, scaler, utility_scaler_path, utility_model_path):
    traces = []
    traces_utilities = []
    current_trace = []
    current_trace_utilities = []
    memory_of_states_visited = []
    possible_actions = generate_all_possible_actions(-30, 30)
    
    while len(traces) < min_num_traces:
    
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
                    
                if len(current_trace) < 10:
                    padding = [[1000,200,1000,200,1000,200]] * (10 - len(current_trace))
                    current_trace = padding + current_trace 
                    padding = [0.001] * (10 - len(current_trace)) 
                    current_trace_utilities = padding + current_trace_utilities
                    
                traces_utilities.append(current_trace_utilities.copy())
                traces.append(current_trace.copy())      
                
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
                intrinsic_utilities = [get_intrinsinc_utility_from_state(predicted_state, memory_of_states_visited, 1) for predicted_state in predicted_states]
                best_action = actions[intrinsic_utilities.index(max(intrinsic_utilities))]
                print("Predicted Red Distance: ", predicted_states[np.argmax(intrinsic_utilities)].get('distance_red'))
                print("Predicted Red Angle: ", predicted_states[np.argmax(intrinsic_utilities)].get('angle_red'))
                print("Predicted utility: ", max(intrinsic_utilities))
            
                print("Best action: ", best_action)
                rob.moveWheelsByTime(best_action[0], best_action[1], 1, wait=True)
                perception_after_action = get_perceptual_state_limited(sim)
                print("Final Red Distance: ", perception_after_action.get('distance_red'))
                print("Final Red Angle: ", perception_after_action.get('angle_red'))
                print("Real utility: ", get_intrinsinc_utility_from_state(perception_after_action, memory_of_states_visited, 1))
                current_state_list = get_array_from_perceptual_dict(perception_after_action)
                memory_of_states_visited.append(current_state_list)
                current_trace.append(current_state_list)
                print("--------------------")
                        
                
            rob.wait(0.1)
            
        if len(memory_of_states_visited) >= 5:
            memory_of_states_visited = memory_of_states_visited[-5:]

    extrinsic_utility_model, extrinsic_utility_scaler = train_model(traces, traces_utilities, utility_scaler_path)
    joblib.dump(extrinsic_utility_model, utility_model_path)

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
                reset_randomize_positions(rob, sim, ["REDCYLINDER", "GREENCYLINDER", "BLUECYLINDER", "CUSTOMCYLINDER"])      
                
                rob.wait(1) 
                rob.wait(1)
                continue
            else:     
                extrinsic_utilities = np.array(get_extrinsic_utility_from_states(predicted_states, extrinsic_utility_model, extrinsic_utility_scaler))          
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
                    reset_randomize_positions(rob, sim, ["REDCYLINDER", "GREENCYLINDER", "BLUECYLINDER", "CUSTOMCYLINDER"])      
                    
                    rob.wait(1) 
                    rob.wait(1)
                    continue
                
                print("-------------")


def test_extrinsic_utility_model(sim, rob, n_rounds, world_model, scaler, extrinsic_utility_model, extrinsic_utility_scaler):
    possible_actions = generate_all_possible_actions(-30, 30)
    round = 0
    while round < n_rounds:
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
                best_action = possible_actions[index_best_action]   
                rob.moveWheelsByTime(best_action[0], best_action[1], 1, wait=True)
                print("Objetive reached")
                round += 1
                sim.wait(1)
                rob.wait(1)
                reset_randomize_positions(rob, sim, ["REDCYLINDER", "GREENCYLINDER", "BLUECYLINDER", "CUSTOMCYLINDER"])      
                
                rob.wait(1) 
                rob.wait(1)
                continue
            else:     
                extrinsic_utilities = np.array(get_extrinsic_utility_from_states(predicted_states, extrinsic_utility_model, extrinsic_utility_scaler))          
                best_action = possible_actions[np.argmax(extrinsic_utilities)]
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
                    round += 1
                    sim.wait(1)
                    rob.wait(1)
                    reset_randomize_positions(rob, sim, ["REDCYLINDER", "GREENCYLINDER", "BLUECYLINDER", "CUSTOMCYLINDER"])      
                    
                    rob.wait(1) 
                    rob.wait(1)
                    continue
                
                print("-------------")
