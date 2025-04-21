import time
import os
import joblib
import numpy as np
import random
from collections import deque

# Robobo / Sim specific imports
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from robobopy.utils.IR import IR

# Imports from your project files
from perceptual_space import get_perceptual_state_limited
# Import NEW utility functions and existing ones
from utils import predict_next_perceptual_state, calculate_novelty, avoid_obstacle

# --- Configuration ---
MODEL_DIR = os.path.join("src", "models")
MODEL_FILENAME = "world_model.joblib"
SCALER_FILENAME = "scaler.joblib"
STATE_MEMORY_SIZE = 50  # How many past states to remember for novelty
ACTION_DURATION_S = 1 # Duration for each chosen motor command (e.g., 1 second)
LOOP_DELAY_S = 0.5 # Small delay between loop iterations

# Define the candidate actions the robot will consider.
POSSIBLE_ACTIONS = [
    (30, 30), (30, 5), (30, 0), (5, 30), (0, 30)
]
# Thresholds from original main.py
FRONT_IR_THRESHOLD = 60
BACK_IR_THRESHOLD = 85

# --- Load Model and Scaler ---
print("Loading world model and scaler...")
try:
    world_model = joblib.load(os.path.join(MODEL_DIR, MODEL_FILENAME))
    scaler = joblib.load(os.path.join(MODEL_DIR, SCALER_FILENAME))
    print("World model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model or scaler not found in {MODEL_DIR}.")
    print("Please train the world model using the notebook first.")
    exit()
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    exit()

# --- Initialize Robot and Simulator ---
print("Initializing Robot and Simulator...")
rob = Robobo("localhost")
sim = RoboboSim("localhost")
sim.connect()
rob.connect()
rob.setEmotionTo(Emotions.HAPPY)
rob.moveTiltTo(100, 50)
print("Initialization Complete.")
time.sleep(1)

# --- Initialize State Memory ---
state_memory = deque(maxlen=STATE_MEMORY_SIZE)

# --- Main Deliberative Loop ---
print("Starting Deliberative Control Loop...")
try:
    while True:
        # 1. Obstacle Check (Reactive safety layer)
        front_ir = rob.readIRSensor(IR.FrontC)
        back_ir = rob.readIRSensor(IR.BackC)

        # *** ADD THIS PRINT STATEMENT ***
        print(f"  IR Sensors Read: Front={front_ir}, Back={back_ir}")
        # *******************************

        if front_ir > FRONT_IR_THRESHOLD or back_ir > BACK_IR_THRESHOLD:
            print(f"Obstacle detected! (Front: {front_ir > FRONT_IR_THRESHOLD}, Back: {back_ir > BACK_IR_THRESHOLD}). Executing avoidance maneuver.")
            avoid_obstacle(rob, front_ir, back_ir)
            # Skip deliberation for this cycle after avoiding
            time.sleep(1) # Wait after avoidance
            continue # Go to the start of the next loop iteration

        # 2. Get Current Perception P(t)
        # (Ensure this is called AFTER the potential 'continue' above)
        current_perception = get_perceptual_state_limited(sim)
        if current_perception is None:
            print("Warning: Failed to get current perception. Skipping cycle.")
            time.sleep(LOOP_DELAY_S)
            continue

        # Store current perception in memory (for novelty)
        state_memory.append(current_perception)
        print(f"\nCurrent State: DistRed={current_perception.get('distance_red', -1):.1f}, AngleRed={current_perception.get('angle_red', -1):.1f}") # Example shorter print

        # 3. Deliberation: Predict outcomes and evaluate actions
        candidate_utilities = {}
        print("Deliberating...")
        for action in POSSIBLE_ACTIONS:
            # Predict P(t+1) using the world model
            predicted_perception = predict_next_perceptual_state(
                current_perception, action, world_model, scaler
            )

            utility = -float('inf') # Default to very low utility

            if predicted_perception is not None:
                # Evaluate the predicted state using Novelty utility model
                utility = calculate_novelty(predicted_perception, state_memory)
                # Make print shorter for clarity during debugging
                print(f"  Action {action} -> Pred DRed: {predicted_perception.get('distance_red', -1):.1f}, Novelty: {utility:.4f}")

            else:
                print(f"  Action {action} -> Prediction failed.")

            candidate_utilities[action] = utility

        # 4. Choose Best Action (highest utility)
        if not candidate_utilities or not any(np.isfinite(list(candidate_utilities.values()))): # Check if any valid utilities exist
             print("Warning: No valid actions could be evaluated. Stopping.")
             best_action = (0, 0) # Stop if deliberation fails
        else:
            # Filter out -inf utilities before finding max, if any prediction failed
             valid_utilities = {a: u for a, u in candidate_utilities.items() if np.isfinite(u)}
             if not valid_utilities:
                 print("Warning: All actions resulted in invalid utility. Stopping.")
                 best_action = (0,0)
             else:
                 best_action = max(valid_utilities, key=valid_utilities.get)
                 max_utility = valid_utilities[best_action]
                 print(f"==> Chosen Action: {best_action} (Utility: {max_utility:.4f})")

        # 5. Execute the chosen best action
        print(f"Executing Action: {best_action}") # Added print before execution
        rSpeed, lSpeed = best_action
        rob.moveWheelsByTime(int(rSpeed), int(lSpeed), ACTION_DURATION_S, wait=True)
        # rob.wait(0.1) # Optional pause

        # 6. Loop Delay
        time.sleep(LOOP_DELAY_S)

except KeyboardInterrupt:
    print("\nStopping control loop.")

finally:
    # --- Cleanup ---
    print("Disconnecting...")
    # Wrap stopMotors in a try-except in case connection is already lost
    try:
        rob.stopMotors()
        rob.moveTiltTo(90, 10) # Park tilt
        rob.setEmotionTo(Emotions.SAD)
    except Exception as e_stop:
        print(f"  Error during cleanup stop: {e_stop}")
    finally:
        # Ensure disconnection happens
        try:
            rob.disconnect()
        except Exception as e_disc_rob:
            print(f"  Error disconnecting robot: {e_disc_rob}")
        try:
            sim.disconnect()
        except Exception as e_disc_sim:
            print(f"  Error disconnecting simulator: {e_disc_sim}")
        print("Disconnected.")