from robobopy.Robobo import Robobo
import random
import math
import numpy as np
from collections import deque

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