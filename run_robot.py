import time
import os
import joblib
import numpy as np
from collections import deque

# Robobo / Sim specific imports
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from robobopy.utils.IR import IR

# Imports from your project files
from perceptual_space import get_perceptual_state_limited
# We don't need perform_random_action, but we might need specific action execution
# If you create a function in action_space.py like `execute_action(rob, rSpeed, lSpeed, duration_ms)` use that.
# Otherwise, call rob.moveWheelsByTime directly.
# from action_space import execute_action # Example if you create this function

# Import NEW utility functions and existing ones
from utils import predict_next_perceptual_state, calculate_novelty, avoid_obstacle

# --- Configuration ---
MODEL_DIR = os.path.join("src", "models")
MODEL_FILENAME = "world_model.joblib"
SCALER_FILENAME = "scaler.joblib"
STATE_MEMORY_SIZE = 50  # How many past states to remember for novelty
ACTION_DURATION_MS = 1 # Duration for each chosen motor command (e.g., 1 second)
LOOP_DELAY_S = 0.2 # Small delay between loop iterations

# Define the candidate actions the robot will consider.
# These should be (rSpeed, lSpeed) tuples. Use values similar to your data collection.
# Example based on limited actions in action_space.py:
POSSIBLE_ACTIONS = [
    (30, 30),  # Forward (approx 0 deg turn)
    (30, 5),   # Slight Right (approx 45 deg turn - depends on calibration/duration)
    (30, 0),   # Hard Right (approx 90 deg turn - depends on calibration/duration)
    (5, 30),   # Slight Left (approx -45 deg turn - depends on calibration/duration)
    (0, 30),   # Hard Left (approx -90 deg turn - depends on calibration/duration)
    #(0, 0)    # Optional: Consider stopping as an action
]

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
rob.setEmotionTo(Emotions.HAPPY) # Let's make it happy!
rob.moveTiltTo(100, 50) # Point camera forward/down
print("Initialization Complete.")
time.sleep(1) # Allow connections to establish

# --- Initialize State Memory ---
state_memory = deque(maxlen=STATE_MEMORY_SIZE)

# --- Main Deliberative Loop ---
print("Starting Deliberative Control Loop...")
try:
    while True:
        # 1. Obstacle Check (Reactive safety layer)
        front_ir = rob.readIRSensor(IR.FrontC)
        back_ir = rob.readIRSensor(IR.BackC)
        # Use thresholds similar to your data collection script
        if front_ir > 60 or back_ir > 85:
            print("Obstacle detected! Executing avoidance maneuver.")
            avoid_obstacle(rob, front_ir, back_ir)
            # Skip deliberation for this cycle after avoiding
            time.sleep(1) # Wait after avoidance
            continue

        # 2. Get Current Perception P(t)
        current_perception = get_perceptual_state_limited(sim)
        if current_perception is None:
            print("Warning: Failed to get current perception. Skipping cycle.")
            time.sleep(LOOP_DELAY_S)
            continue

        # Store current perception in memory (for novelty)
        # Important: store *before* executing the chosen action
        state_memory.append(current_perception)
        print(f"\nCurrent State: {current_perception}")

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
                print(f"  Action {action} -> Predicted State (e.g., RedDist): {predicted_perception.get('distance_red', -1):.1f}, Novelty: {utility:.4f}")

                # (Future Extension: Add goal utility check here if goal found)
                # (Future Extension: Add obstacle prediction check here based on predicted state)

            else:
                print(f"  Action {action} -> Prediction failed.")

            candidate_utilities[action] = utility

        # 4. Choose Best Action (highest utility)
        if not candidate_utilities:
            print("Warning: No actions could be evaluated. Stopping.")
            best_action = (0, 0) # Stop if deliberation fails
        else:
            # Find the action tuple associated with the maximum utility
            best_action = max(candidate_utilities, key=candidate_utilities.get)
            max_utility = candidate_utilities[best_action]
            print(f"==> Chosen Action: {best_action} (Utility: {max_utility:.4f})")

        # 5. Execute the chosen best action
        rSpeed, lSpeed = best_action
        rob.moveWheelsByTime(int(rSpeed), int(lSpeed), ACTION_DURATION_MS, wait=True)
        # Adding a small pause after movement can sometimes help sim stability
        rob.wait(0.1)

        # 6. Loop Delay
        time.sleep(LOOP_DELAY_S)

except KeyboardInterrupt:
    print("\nStopping control loop.")

finally:
    # --- Cleanup ---
    print("Disconnecting...")
    rob.stopMotors()
    rob.moveTiltTo(90, 10) # Park tilt
    rob.setEmotionTo(Emotions.SAD)
    rob.disconnect()
    sim.disconnect()
    print("Disconnected.")