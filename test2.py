import time
import os
import joblib
import numpy as np
import random  # Make sure random is imported
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
STATE_MEMORY_SIZE = 50
# *** ADJUSTED ACTION DURATION ***
ACTION_DURATION_S = 1 # Duration in SECONDS (use float if library supports, else int(1))
# *** INCREASED LOOP DELAY ***
LOOP_DELAY_S = 0.5 # Increase delay to reduce message frequency (try 0.5s first)
# ***************************
EXPLORATION_RATE_EPSILON = 0.1 # 10% chance of random action

# Define the candidate actions
POSSIBLE_ACTIONS = [
    (30, 30), (30, 5), (30, 0), (5, 30), (0, 30)
]
# Thresholds
FRONT_IR_THRESHOLD = 60 # Keep original for now, adjust if needed
BACK_IR_THRESHOLD = 85

# --- Load Model and Scaler ---
# (Loading code remains the same)
print("Loading world model and scaler...")
try:
    world_model = joblib.load(os.path.join(MODEL_DIR, MODEL_FILENAME))
    scaler = joblib.load(os.path.join(MODEL_DIR, SCALER_FILENAME))
    print("World model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model or scaler not found in {MODEL_DIR}.")
    exit()
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    exit()

# --- Initialize Robot and Simulator ---
# (Initialization code remains the same)
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
        print(f"  IR Sensors Read: Front={front_ir}, Back={back_ir}")

        if front_ir > FRONT_IR_THRESHOLD or back_ir > BACK_IR_THRESHOLD:
            print(f"Obstacle detected! (Front: {front_ir > FRONT_IR_THRESHOLD}, Back: {back_ir > BACK_IR_THRESHOLD}). Executing avoidance maneuver.")
            avoid_obstacle(rob, front_ir, back_ir)
            time.sleep(1) # Wait after avoidance
            continue # Go to the start of the next loop iteration

        # 2. Get Current Perception P(t)
        current_perception = get_perceptual_state_limited(sim)
        if current_perception is None:
            print("Warning: Failed to get current perception. Skipping cycle.")
            time.sleep(LOOP_DELAY_S) # Still delay even if perception fails
            continue

        # Store current perception BEFORE executing action
        # Check if state_memory needs pruning or if deque handles it
        state_memory.append(current_perception)
        print(f"\nCurrent State: DistRed={current_perception.get('distance_red', -1):.1f}, AngleRed={current_perception.get('angle_red', -1):.1f}")

        # 3. Deliberation: Predict outcomes and evaluate actions
        candidate_utilities = {}
        print("Deliberating...")
        for action in POSSIBLE_ACTIONS:
            predicted_perception = predict_next_perceptual_state(
                current_perception, action, world_model, scaler
            )
            utility = -float('inf')
            if predicted_perception is not None:
                utility = calculate_novelty(predicted_perception, state_memory)
                print(f"  Action {action} -> Pred DRed: {predicted_perception.get('distance_red', -1):.1f}, Novelty: {utility:.4f}")
            else:
                print(f"  Action {action} -> Prediction failed.")
            candidate_utilities[action] = utility

        # 4. Choose Best Action (Epsilon-Greedy)
        if not candidate_utilities or not any(np.isfinite(list(candidate_utilities.values()))):
            print("Warning: No valid actions could be evaluated. Setting action to stop.")
            best_action_nominal = (0, 0)
        else:
            valid_utilities = {a: u for a, u in candidate_utilities.items() if np.isfinite(u)}
            if not valid_utilities:
                print("Warning: All actions resulted in invalid utility. Setting action to stop.")
                best_action_nominal = (0,0)
            else:
                best_action_nominal = max(valid_utilities, key=valid_utilities.get)
                max_utility = valid_utilities[best_action_nominal]
                print(f"  Best Calculated Action: {best_action_nominal} (Utility: {max_utility:.4f})")

        # Epsilon-greedy selection
        if random.random() < EXPLORATION_RATE_EPSILON:
            best_action = random.choice(POSSIBLE_ACTIONS)
            print(f"==> Exploring: Randomly chose Action: {best_action}")
        else:
            best_action = best_action_nominal
            print(f"==> Exploiting: Chose Best Calculated Action: {best_action}")

        # 5. FINAL Obstacle Check (Optional - keep first check or add this redundant one)
        # execute_this_action = best_action
        # action_overridden = False
        # front_ir_exec = rob.readIRSensor(IR.FrontC) # Re-read just before exec
        # back_ir_exec = rob.readIRSensor(IR.BackC)
        # if (best_action[0] > 0 or best_action[1] > 0) and front_ir_exec > FRONT_IR_THRESHOLD:
        #     print(f"Obstacle detected ahead ({front_ir_exec}) just before execution! Overriding.")
        #     avoid_obstacle(rob, front_ir_exec, back_ir_exec)
        #     action_overridden = True
        # # Add back IR check if needed
        # elif (best_action[0] < 0 or best_action[1] < 0) and back_ir_exec > BACK_IR_THRESHOLD:
        #     # ... avoidance ...
        #     action_overridden = True

        # 6. Execute Action (only if not overridden - if using second obstacle check)
        # if not action_overridden:
        # Execute chosen action (best_action from epsilon-greedy)
        print(f"Executing Action: {best_action} for {ACTION_DURATION_S} second(s)")
        rSpeed, lSpeed = best_action
        # Check if duration needs to be int or can be float
        try:
            # Try passing float first if allowed
            rob.moveWheelsByTime(int(rSpeed), int(lSpeed), float(ACTION_DURATION_S), wait=True)
        except TypeError:
            # Fallback to int if float fails
            rob.moveWheelsByTime(int(rSpeed), int(lSpeed), int(ACTION_DURATION_S), wait=True)
        # else:
        #     print("Skipping chosen action due to final obstacle check.")
        #     time.sleep(0.5)

        # 7. Loop Delay
        time.sleep(LOOP_DELAY_S) # Now waits 0.5 seconds between cycles

except KeyboardInterrupt:
    print("\nStopping control loop.")

finally:
    # (Cleanup code as before)
    # ...
    print("Disconnected.")