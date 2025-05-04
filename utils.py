from robobopy.Robobo import Robobo
import random
import math
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional, Any

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

# --- World Model Prediction Function (Continous) ---
def predict_multiple_next_states(
    current_perception: Dict[str, float],
    candidate_actions: List[Tuple[int, int]],
    model: Any, # Replace Any with specific model type if known (e.g., sklearn model)
    scaler: Any # Replace Any with specific scaler type
) -> Tuple[List[Optional[Dict[str, float]]], List[Tuple[int, int]]]:
    """
    Predicts the next perceptual states for multiple candidate actions using the loaded world model.

    Args:
        current_perception (dict): The current perception state
                                   (e.g., {'distance_red': d, 'angle_red': a, ...}).
        candidate_actions (list): A list of candidate actions, where each action
                                  is a tuple (rSpeed, lSpeed).
        model: The loaded world model (e.g., from joblib.load).
        scaler: The loaded scaler (e.g., from joblib.load).

    Returns:
        tuple: A tuple containing two lists:
            1. predicted_states (list): A list of predicted next state dictionaries.
                                        Each element corresponds to an action in
                                        candidate_actions. Contains None if prediction
                                        failed for a specific action.
            2. actions_used (list): The list of candidate actions that were processed.
                                    This list corresponds index-wise to predicted_states.
    """
    predicted_states: List[Optional[Dict[str, float]]] = []
    actions_used: List[Tuple[int, int]] = [] # Keep track of actions corresponding to predictions

    # --- Validate current_perception once before the loop ---
    if not isinstance(current_perception, dict):
        print("Error: Invalid current_perception format.")
        # Return empty lists as specified by the return type hint
        return [], []

    # --- Define the exact order of perception keys used during training ---
    # This MUST match the order of columns 1-6 in your training data
    perception_keys_ordered = [
        'distance_red', 'angle_red',
        'distance_green', 'angle_green',
        'distance_blue', 'angle_blue'
    ]

    # --- Loop through each candidate action ---
    for action in candidate_actions:
        actions_used.append(action) # Store the action we are processing

        # Optional: Validate individual action format if needed, though
        # generate_candidate_actions should ensure this.
        if not isinstance(action, (tuple, list)) or len(action) != 2:
            print(f"Warning: Skipping invalid action format in candidate list: {action}")
            predicted_states.append(None) # Maintain list correspondence
            continue

        try:
            # --- Convert input perception dict to the ordered list ---
            # Use .get(key, 0.0) for safety if a key might be missing, assuming 0 is a safe default
            current_state_list = [current_perception.get(key, 0.0) for key in perception_keys_ordered]

            # Combine initial state and action into the feature vector
            # The order MUST match the columns 1-8 used for training the scaler/model
            feature_vector_list = current_state_list + list(action)
            feature_vector = np.array(feature_vector_list).reshape(1, -1) # Reshape for single sample prediction

            # --- Scale the input feature vector ---
            feature_vector_scaled = scaler.transform(feature_vector)

            # --- Predict the next state ---
            # model.predict is expected to return an array of shape (1, num_output_features)
            predicted_next_state_array = model.predict(feature_vector_scaled)[0] # Get the prediction for this action

            # --- Convert prediction array back to dictionary format ---
            # The order MUST match the columns 9-14 used as targets during training
            predicted_perception_dict = {
                key: value for key, value in zip(perception_keys_ordered, predicted_next_state_array)
            }
            predicted_states.append(predicted_perception_dict)

        except Exception as e:
            print(f"Error during prediction for action {action}: {e}")
            predicted_states.append(None) # Append None if prediction fails for this action

    # --- Return the corresponding lists ---
    return predicted_states, actions_used

# --- World Model Prediction Function (BATCHED Version) ---
def predict_multiple_next_states_batched(
    current_perception: Dict[str, float],
    candidate_actions: List[Tuple[int, int]],
    model: Any, # Replace Any with specific model type if known
    scaler: Any # Replace Any with specific scaler type
) -> Tuple[List[Optional[Dict[str, float]]], List[Tuple[int, int]]]:
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
    predicted_states: List[Optional[Dict[str, float]]] = []
    actions_used: List[Tuple[int, int]] = list(candidate_actions) # Copy actions upfront

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


# --- Novelty Calculation Function ---
def calculate_novelty(predicted_state, memory, n=1):
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
    
# --- Utility Calculation Function ---
def calculate_extrinsic_utility(predicted_state: dict) -> float:
    """
    Calculates the utility of a state based on proximity to the red cylinder.
    Higher utility means closer to the goal.
    This is a basic placeholder - should be replaced by learned ANN later.
    """
    if not predicted_state or 'distance_red' not in predicted_state:
        print("Warning: Invalid predicted state for utility calculation.")
        return -1000.0 # Very low utility if state is invalid or lacks goal info

    distance_to_red = predicted_state['distance_red']

    # Avoid division by zero or issues with non-positive distances
    if distance_to_red <= 0.1: # Use a small positive threshold instead of 0
        return 1000.0 # Assign high utility if very close or distance is non-positive

    # Example: Utility inversely proportional to distance
    # Closer is better -> higher utility
    utility = 100.0 / distance_to_red # Scaled inverse distance

    # --- Optional refinements (Add later if needed) ---
    # # Small penalty for being close to other cylinders
    # distance_green = predicted_state.get('distance_green', 10000) # Default far if not present
    # distance_blue = predicted_state.get('distance_blue', 10000) # Default far if not present
    # utility -= 10.0 / max(distance_green, 0.1) # Penalty increases sharply when close
    # utility -= 10.0 / max(distance_blue, 0.1)

    # # Reward for facing the target (Optional, needs angle calculation)
    # angle_to_red = predicted_state.get('angle_red', 180) # Default opposite if not present
    # # Reward smaller absolute angles (facing the target is angle 0)
    # utility += 10.0 * (1 - abs(angle_to_red) / 180.0)

    return utility