from src.perceptual_space import get_perceptual_state_limited
import numpy as np
from utils.world_model_utils import load_world_model, load_scaler, get_possible_action

def predict_next_perceptual_state(current_perception, action, model, scaler):

    try:
        current_state_list = [
            current_perception.get('distance_red', 0), current_perception.get('angle_red', 0),
            current_perception.get('distance_green', 0), current_perception.get('angle_green', 0),
            current_perception.get('distance_blue', 0), current_perception.get('angle_blue', 0)
        ]

        feature_vector_list = current_state_list + list(action)
        feature_vector = np.array(feature_vector_list).reshape(1, -1) # Reshape for single sample

        feature_vector_scaled = scaler.transform(feature_vector)

        predicted_next_state_array = model.predict(feature_vector_scaled)[0]

        predicted_perception = {
            'distance_red': predicted_next_state_array[0], 'angle_red': predicted_next_state_array[1],
            'distance_green': predicted_next_state_array[2], 'angle_green': predicted_next_state_array[3],
            'distance_blue': predicted_next_state_array[4], 'angle_blue': predicted_next_state_array[5]
        }
        return predicted_perception

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    
def predict_multiple_next_states_batched(
    current_perception: dict,
    candidate_actions: list,
    model,
    scaler 
):
    predicted_states = []
    actions_used = list(candidate_actions.copy())

    if not isinstance(current_perception, dict):
        print("Error: Invalid current_perception format.")
        return [None] * len(actions_used), actions_used
    
    perception_keys_ordered = [
        'distance_red', 'angle_red',
        'distance_green', 'angle_green',
        'distance_blue', 'angle_blue'
    ]
    num_perception_features = len(perception_keys_ordered)
    num_action_features = 2

    try:
        batch_feature_vectors = []
        current_state_list = [current_perception.get(key, 0.0) for key in perception_keys_ordered]

        for action in candidate_actions:
            if not isinstance(action, (tuple, list)) or len(action) != num_action_features:
                print(f"Warning: Skipping invalid action format: {action}")
                feature_vector_list = [0.0] * (num_perception_features + num_action_features)
            else:
                 feature_vector_list = current_state_list + list(action)

            batch_feature_vectors.append(feature_vector_list)

        batch_np = np.array(batch_feature_vectors, dtype=np.float64)
        batch_scaled = scaler.transform(batch_np)
        predictions_batch_np = model.predict(batch_scaled)

        for i, predicted_next_state_array in enumerate(predictions_batch_np):
            original_action = candidate_actions[i]
            if not isinstance(original_action, (tuple, list)) or len(original_action) != num_action_features:
                predicted_states.append(None) # Ensure None for invalid original actions
                continue

            predicted_perception_dict = {
                key: value for key, value in zip(perception_keys_ordered, predicted_next_state_array)
            }
            predicted_states.append(predicted_perception_dict)

    except Exception as e:
        print(f"Error during batch prediction/scaling: {e}")
        return [None] * len(actions_used), actions_used

    return predicted_states, actions_used

def test_world_model(rob, sim, world_model_path, scaler_path):
    world_model = load_world_model(world_model_path)
    scaler = load_scaler(scaler_path)
    rSpeed, lSpeed = get_possible_action(-30,30)
    initial_perception = get_perceptual_state_limited(sim)
    predicted_perception = predict_next_perceptual_state(initial_perception, (rSpeed, lSpeed), world_model, scaler)
    rob.moveWheelsByTime(rSpeed, lSpeed, 1, wait=True)
    final_perception = get_perceptual_state_limited(sim)
    print(" Initial perception: ", initial_perception)
    print(" Action: ", (rSpeed, lSpeed))
    print(" Predicted perception: ", predicted_perception)
    print(" Final perception: ", final_perception)
    predicted_perception_array = np.array([predicted_perception[k] for k in predicted_perception.keys()], dtype=np.float64)
    final_perception_array = np.array([final_perception[k] for k in final_perception.keys()], dtype=np.float64)
    mae = np.mean(np.abs(predicted_perception_array - final_perception_array))
    mse = np.mean((predicted_perception_array - final_perception_array)**2)
    rmse = np.sqrt(mse)
    print(" MAE: ", mae, ", MSE: ", mse, ", RMSE: ", rmse)
    
    