from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from perceptual_space import get_perceptual_state, get_perceptual_state_limited
import time
from robobopy.utils.IR import IR
import joblib
import random
from utils import predict_next_perceptual_state
import numpy as np

def load_world_model(world_model_path: str):
    return joblib.load(world_model_path)

def load_scaler(scaler_path: str):
    return joblib.load(scaler_path)

def get_possible_action(under_limit: int, upper_limit: int):
    rSpeed = random.randint(under_limit, upper_limit)
    lSpeed = random.randint(under_limit, upper_limit)
    return rSpeed, lSpeed

def test_world_model(rob, sim):
    world_model = load_world_model("src/models/world_model_v3.joblib")
    scaler = load_scaler("src/models/scaler_v3.joblib")
    rSpeed, lSpeed = get_possible_action(-30, 30)
    initial_perception = get_perceptual_state_limited(sim)
    predicted_perception = predict_next_perceptual_state(initial_perception, (rSpeed, lSpeed), world_model, scaler)
    rob.moveWheelsByTime(rSpeed, lSpeed, 1, wait=True)
    final_perception = get_perceptual_state_limited(sim)

    print("\n--- Test World Model ---")
    print(f"Initial Perception: {initial_perception}\n")
    print(f"Action: (rSpeed={rSpeed}, lSpeed={lSpeed})\n")
    print(f"Predicted Perception: {predicted_perception}\n")
    print(f"Final Perception: {final_perception}")

    predicted_perception_array = np.array([predicted_perception[k] for k in predicted_perception.keys()], dtype=np.float64)
    final_perception_array = np.array([final_perception[k] for k in final_perception.keys()], dtype=np.float64)
    mae = np.mean(np.abs(predicted_perception_array - final_perception_array))
    mse = np.mean((predicted_perception_array - final_perception_array) ** 2)
    rmse = np.sqrt(mse)

    print("\n--- Metrics ---")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("-----------------------\n")
    
    