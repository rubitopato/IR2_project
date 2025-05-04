from robobopy.Robobo import Robobo
import random
import math
import numpy as np
from collections import deque
import joblib
from scipy.spatial.distance import euclidean

def avoid_obstacle(rob, frontal_distance, back_distance):
    rob.stopMotors()
    if frontal_distance > back_distance:
        rob.moveWheelsByTime(-20,-20,2, wait=True)
        rob.moveWheelsByTime(-10,10,2, wait=True)
    else:
        rob.moveWheelsByTime(20,20,2, wait=True)
        rob.moveWheelsByTime(-10,10,2, wait=True)

def get_cylinders_initial_pos(sim, objects):
    cylinders_initial_pos = {
        'REDCYLINDER': None,
        'GREENCYLINDER': None,
        'BLUECYLINDER': None,
        'CUSTOMCYLINDER': None
    }

    if objects != None and len(objects) > 0:
        for object in objects:
            location = sim.getObjectLocation(object)
            if object == 'REDCYLINDER' or object == 'GREENCYLINDER' or object == 'BLUECYLINDER':  # Determine the cylinders position
                cylinders_initial_pos[object] = location['position']
            elif object == 'CUSTOMCYLINDER':  # Move the cylinder to one corner of the simulation
                sim.setObjectLocation('CUSTOMCYLINDER',{'x': -900.0, 'y': 10.0, 'z': 950.0})
    return cylinders_initial_pos


def move_cylinder(sim, cylinder_name):  # Move the cylinder +50 in the X axis (test)
    loc = sim.getObjectLocation(cylinder_name)
    pos = loc['position']
    pos["x"] += random.randint(-150, 150)
    pos["z"] += random.randint(-150, 150)
    sim.setObjectLocation(cylinder_name, pos)

def reset_position_cylinders(sim, cylinders_initial_pos):
    for cylinder_name, initial_position in cylinders_initial_pos.items():
        sim.setObjectLocation(cylinder_name, initial_position)
        
def save_new_line_of_data(perception_init, rSpeed, lSpeed, perception_final):
    line = f'{perception_init["distance_red"]},{perception_init["angle_red"]},{perception_init["distance_green"]},{perception_init["angle_green"]},'
    line += f'{perception_init["distance_blue"]},{perception_init["angle_blue"]},{rSpeed},{lSpeed},'
    line += f'{perception_final["distance_red"]},{perception_final["angle_red"]},{perception_final["distance_green"]},{perception_final["angle_green"]},'
    line += f'{perception_final["distance_blue"]},{perception_final["angle_blue"]}'
    with open("dataset/dataset_limited.txt", "a") as archivo:
        archivo.write(f"\n{line}")

# --- World Model Prediction Function ---
def predict_next_perceptual_state(current_perception, action, model, scaler):
    """
    Predicts the next perceptual state using the loaded world model.

    Args:
        current_perception (dict): The current perception state
                                   (e.g., {'distance_red': d, 'angle_red': a, ...}).
        action (tuple): The action taken (rSpeed, lSpeed).
        model: The loaded world model (e.g., from joblib.load).
        scaler: The loaded scaler (e.g., from joblib.load).

    Returns:
        dict: The predicted next perceptual state dictionary.
              Returns None if prediction fails or input is invalid.
    """
    if not isinstance(current_perception, dict) or not isinstance(action, (tuple, list)) or len(action) != 2:
         print("Error: Invalid input format for prediction.")
         return None

    try:
        # --- Convert input perception dict to the ordered list/array format ---
        # Ensure the order matches the first 6 columns used for training
        current_state_list = [
            current_perception.get('distance_red', 0), current_perception.get('angle_red', 0),
            current_perception.get('distance_green', 0), current_perception.get('angle_green', 0),
            current_perception.get('distance_blue', 0), current_perception.get('angle_blue', 0)
        ]

        # Combine initial state and action
        feature_vector_list = current_state_list + list(action)
        feature_vector = np.array(feature_vector_list).reshape(1, -1) # Reshape for single sample

        # --- Scale the input feature vector ---
        feature_vector_scaled = scaler.transform(feature_vector)

        # --- Predict the next state ---
        predicted_next_state_array = model.predict(feature_vector_scaled)[0] # Get the first (only) prediction

        # --- Convert prediction back to dictionary format ---
        # Ensure the order matches the last 6 columns used for training
        predicted_perception = {
            'distance_red': predicted_next_state_array[0], 'angle_red': predicted_next_state_array[1],
            'distance_green': predicted_next_state_array[2], 'angle_green': predicted_next_state_array[3],
            'distance_blue': predicted_next_state_array[4], 'angle_blue': predicted_next_state_array[5]
        }
        return predicted_perception

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# --- Novelty Calculation Function ---
def get_intrinsinc_utility_from_state(candidate_state, memory_states, n=1.0):
    vector = get_array_from_perceptual_dict(candidate_state)
    if not memory_states:
        return float('inf')  # máxima novedad si no hay memoria
    distances = [euclidean(vector, mem_state) for mem_state in memory_states]
    novelty = np.mean([d**n for d in distances])
    return novelty

def novelty_score(predicted_state, memory, n=1):
    """
    Calculates the novelty of a predicted state based on past states in memory.

    Args:
        predicted_state (dict): The predicted state dictionary.
        memory (deque): A deque containing past state dictionaries.
        n (int): Coefficient for distance calculation.

    Returns:
        float: The novelty score. Returns a high value if memory is empty.
    """
    if not memory:
        return 1000.0 # Return a high novelty score if memory is empty

    if not isinstance(predicted_state, dict):
        print("Error: predicted_state must be a dictionary.")
        return 0.0

    try:
        # Convert predicted state dict to a comparable list/tuple (use a consistent order)
        pred_vector = np.array([
            predicted_state.get('distance_red', 0), predicted_state.get('angle_red', 0),
            predicted_state.get('distance_green', 0), predicted_state.get('angle_green', 0),
            predicted_state.get('distance_blue', 0), predicted_state.get('angle_blue', 0)
        ])

        total_distance = 0.0
        num_valid_past = 0
        for past_state in memory:
            if isinstance(past_state, dict):
                # Convert past state dict to a vector
                past_vector = np.array([
                    past_state.get('distance_red', 0), past_state.get('angle_red', 0),
                    past_state.get('distance_green', 0), past_state.get('angle_green', 0),
                    past_state.get('distance_blue', 0), past_state.get('angle_blue', 0)
                ])
                # Calculate Euclidean distance
                dist = np.linalg.norm(pred_vector - past_vector)
                total_distance += dist ** n
                num_valid_past += 1
            else:
                print("Warning: Item in memory is not a dictionary, skipping.")


        if num_valid_past == 0:
             return 1000.0 # High novelty if no valid past states to compare

        # Calculate average distance (novelty score)
        novelty = total_distance / num_valid_past
        return novelty

    except Exception as e:
        print(f"Error during novelty calculation: {e}")
        return 0.0 # Return low novelty on error
    
def objective_found(perception: dict):   
    distance_red = perception.get('distance_red')
    return distance_red < 200

def get_array_from_perceptual_dict(perception: dict):
    return [
        perception.get('distance_red', 0), perception.get('angle_red', 0),
        perception.get('distance_green', 0), perception.get('angle_green', 0),
        perception.get('distance_blue', 0), perception.get('angle_blue', 0)
    ]
    
def predict_multiple_next_states_batched(
    current_perception: dict,
    candidate_actions: list,
    model,
    scaler 
):
    """
    Predicts the next perceptual states for multiple candidate actions using batch processing.

    Args:
        current_perception (dict): The current perception state.
        candidate_actions (list): A list of candidate actions (rSpeed, lSpeed tuples).
        model: The loaded world model (assumed to handle batch prediction).
        scaler: The loaded scaler (assumed to handle batch transformation).

    Returns:
        tuple: A tuple containing two lists:
            1. predicted_states (list): List of predicted next state dictionaries (or None on error).
            2. actions_used (list): The list of candidate actions processed.
    """
    predicted_states = []
    actions_used = list(candidate_actions.copy()) # Copy actions upfront

    # --- Validate inputs ---
    if not isinstance(current_perception, dict):
        print("Error: Invalid current_perception format.")
        return [None] * len(actions_used), actions_used # Return list of Nones

    if not candidate_actions:
        return [], [] # Return empty lists if no actions provided

    # --- Define the exact order of perception keys used during training ---
    # This MUST match the order of columns 1-6 in your training data
    perception_keys_ordered = [
        'distance_red', 'angle_red',
        'distance_green', 'angle_green',
        'distance_blue', 'angle_blue'
    ]
    num_perception_features = len(perception_keys_ordered)
    num_action_features = 2
    num_output_features = num_perception_features # Assuming model predicts the next state directly

    try:
        # --- Prepare the batch input ---
        batch_feature_vectors = []
        # Extract current state values once
        current_state_list = [current_perception.get(key, 0.0) for key in perception_keys_ordered]

        for action in candidate_actions:
            if not isinstance(action, (tuple, list)) or len(action) != num_action_features:
                print(f"Warning: Skipping invalid action format: {action}")
                # We'll handle this later when constructing the final list if needed,
                # but ideally, the input candidate_actions list is clean.
                # For simplicity in batching, we proceed assuming valid actions for now,
                # or raise an error if strictness is required.
                # Let's create a placeholder row of zeros to maintain array shape,
                # but mark its prediction as None later.
                feature_vector_list = [0.0] * (num_perception_features + num_action_features) # Placeholder
            else:
                 # Combine initial state and action
                 # The order MUST match the columns 1-8 used for training scaler/model
                 feature_vector_list = current_state_list + list(action)

            batch_feature_vectors.append(feature_vector_list)

        # Convert the list of lists into a 2D NumPy array
        batch_np = np.array(batch_feature_vectors, dtype=np.float64)

        if batch_np.shape[0] == 0: # If all actions were invalid
             return [None] * len(actions_used), actions_used

        # --- Scale the entire batch ---
        # Assumes scaler.transform accepts and returns a 2D NumPy array
        batch_scaled = scaler.transform(batch_np)

        # --- Predict on the entire scaled batch ---
        # Assumes model.predict accepts a 2D NumPy array and returns a 2D NumPy array
        # where each row is the prediction for the corresponding input row.
        predictions_batch_np = model.predict(batch_scaled)

        # --- Validate prediction shape ---
        if predictions_batch_np.shape[0] != len(candidate_actions):
             print(f"Error: Prediction output rows ({predictions_batch_np.shape[0]}) "
                   f"do not match number of actions ({len(candidate_actions)}).")
             return [None] * len(actions_used), actions_used

        if predictions_batch_np.shape[1] != num_output_features:
             print(f"Error: Prediction output columns ({predictions_batch_np.shape[1]}) "
                   f"do not match expected features ({num_output_features}).")
             return [None] * len(actions_used), actions_used


        # --- Convert predictions back to dictionary format ---
        for i, predicted_next_state_array in enumerate(predictions_batch_np):
             # Check if the original action for this index was valid
             original_action = candidate_actions[i]
             if not isinstance(original_action, (tuple, list)) or len(original_action) != num_action_features:
                 predicted_states.append(None) # Ensure None for invalid original actions
                 continue

             # Convert the prediction row (NumPy array) to a dictionary
             predicted_perception_dict = {
                 key: value for key, value in zip(perception_keys_ordered, predicted_next_state_array)
             }
             predicted_states.append(predicted_perception_dict)

    except Exception as e:
        print(f"Error during batch prediction/scaling: {e}")
        # If any error occurs during batch processing, return Nones for all
        return [None] * len(actions_used), actions_used

    # --- Return the corresponding lists ---
    # actions_used was populated at the start
    return predicted_states, actions_used

def load_world_model(world_model_path: str):
    return joblib.load(world_model_path)

def load_scaler(scaler_path: str):
    return joblib.load(scaler_path)