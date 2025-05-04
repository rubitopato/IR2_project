# Conceptual main loop in main.py or a new script

from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.Emotions import Emotions
from robobopy.utils.IR import IR
from perceptual_space import get_perceptual_state_limited
from action_space import generate_candidate_actions, generate_all_possible_actions # Import action generation
# Import the *new* prediction function and other utils
from utils import predict_multiple_next_states, avoid_obstacle, calculate_extrinsic_utility, predict_multiple_next_states_batched # Assuming utility is also in utils now
from test_word_model import load_world_model, load_scaler # Reuse loading functions
from collections import deque # If you plan to use memory for novelty or learning
import time
import numpy as np # For argmax
import random   # For fallback action

# --- Configuration ---
# NUM_CANDIDATE_ACTIONS = 20 # Increased for better choice, adjust as needed
GOAL_THRESHOLD = 50 # Example: distance to red cylinder considered "goal reached"
OBSTACLE_THRESHOLD_FRONT = 60
OBSTACLE_THRESHOLD_BACK = 85 # Adjust if Back IR is unreliable or absent
ACTION_DURATION = 1 # Seconds to execute the chosen action

# --- Initialization ---
rob = Robobo("localhost")
sim = RoboboSim("localhost")
rob.connect()
sim.connect()

rob.setEmotionTo(Emotions.ANGRY) # Indicate goal-seeking
rob.moveTiltTo(100, 50)

# Load the learned world model and scaler
try:
    world_model = load_world_model("src/models/world_model_v3.joblib")
    scaler = load_scaler("src/models/scaler_v3.joblib")
    print("World model and scaler loaded successfully.")
except FileNotFoundError:
    print("ERROR: World model or scaler file not found. Exiting.")
    rob.disconnect()
    sim.disconnect()
    exit()
except Exception as e:
    print(f"ERROR: Could not load model or scaler: {e}. Exiting.")
    rob.disconnect()
    sim.disconnect()
    exit()


# --- Main Deliberative Loop ---
running = True
step_count = 0
MAX_STEPS = 200 # Add a maximum step count to prevent infinite loops

while running and step_count < MAX_STEPS:
    step_count += 1
    print(f"\n--- Step {step_count} ---")

    # 1. Obstacle Avoidance Check (Reactive Layer)
    front_ir = rob.readIRSensor(IR.FrontC)
    # Handle potential missing back sensor gracefully
    try:
        back_ir = rob.readIRSensor(IR.BackC)
    except Exception: # Replace with more specific exception if known
         back_ir = 0 # Assume no obstacle if sensor reading fails
         print("Warning: Could not read Back IR sensor.")

    if front_ir > OBSTACLE_THRESHOLD_FRONT or back_ir > OBSTACLE_THRESHOLD_BACK:
        print(f"Obstacle detected! Front: {front_ir}, Back: {back_ir}. Avoiding...")
        # Make sure avoid_obstacle uses blocking moves or includes waits
        avoid_obstacle(rob, front_ir, back_ir)
        print("Avoidance maneuver complete.")
        time.sleep(0.2) # Short pause after avoidance
        continue # Skip deliberation for this cycle

    # 2. Perception
    try:
        current_perception = get_perceptual_state_limited(sim)
        if not current_perception or 'distance_red' not in current_perception:
             print("Error: Invalid perception received. Skipping step.")
             time.sleep(0.5)
             continue
        print(f"Current Perception: {current_perception}")
    except Exception as e:
        print(f"Error getting perception: {e}. Skipping step.")
        time.sleep(0.5)
        continue


    # Check if goal reached
    if current_perception['distance_red'] < GOAL_THRESHOLD:
        print("Goal Reached!")
        rob.setEmotionTo(Emotions.LAUGHING)
        rob.stopMotors()
        # Potentially save the trace that led here for learning the utility model
        running = False # Stop the loop
        continue

    # 3. Deliberation
    # 3a. Generate Candidate Actions
    # candidate_actions = generate_candidate_actions(NUM_CANDIDATE_ACTIONS)
    print("Generating ALL possible actions...") # Add print statement
    start_gen_time = time.time()
    candidate_actions = generate_all_possible_actions() # New way - VERY SLOW LOOP AHEAD
    print(f"Action generation took: {time.time() - start_gen_time:.4f}s")

    # 3b. Predict Outcomes for all candidates
    print(f"Predicting outcomes for {len(candidate_actions)} actions...")
    start_pred_time = time.time()
    list_of_predicted_states, list_of_actions_used = predict_multiple_next_states(
        current_perception,
        candidate_actions,
        world_model,
        scaler
    )
    print(f"Prediction took: {time.time() - start_pred_time:.4f}s")
    # Note: list_of_actions_used should usually be identical to candidate_actions

    # 3c. Evaluate Utility for each predicted state
    predicted_utilities = []
    print("Evaluating utilities...")
    start_eval_time = time.time()
    for i, predicted_state in enumerate(list_of_predicted_states):
        action = list_of_actions_used[i] # Get the corresponding action
        if predicted_state:
            utility = calculate_extrinsic_utility(predicted_state)
            predicted_utilities.append(utility)
            # print(f"  Action: {action}, Predicted State (Red Dist): {predicted_state.get('distance_red', 'N/A'):.2f}, Utility: {utility:.4f}") # Example Verbose output
        else:
            # Handle prediction failure (already handled by returning None)
            predicted_utilities.append(-float('inf')) # Assign very low utility
            # print(f"  Action: {action}, Prediction Failed") # Verbose
    print(f"Utility evaluation took: {time.time() - start_eval_time:.4f}s")


    # 3d. Select Best Action
    if not predicted_utilities or all(u == -float('inf') for u in predicted_utilities):
        print("Warning: No valid actions or predictions found. Performing random fallback action.")
        # Fallback: perform a random move if deliberation fails
        best_action = random.choice(generate_candidate_actions(1, min_speed=-15, max_speed=15))[0] # Generate one less aggressive random action
        best_utility = -float('inf') # Indicate fallback
    else:
        best_action_index = np.argmax(predicted_utilities)
        best_action = list_of_actions_used[best_action_index]
        best_utility = predicted_utilities[best_action_index]
        print(f"Selected Action: {best_action} (Predicted Utility: {best_utility:.4f})")

    # 4. Action Execution
    rSpeed, lSpeed = best_action
    print(f"Executing action: rSpeed={rSpeed}, lSpeed={lSpeed} for {ACTION_DURATION}s")
    # Ensure the action execution is blocking for the specified duration
    rob.moveWheelsByTime(rSpeed, lSpeed, ACTION_DURATION, wait=True)

    # Optional: Short pause allows simulation physics/robot state to update before next perception
    time.sleep(0.1)


    

    # 5. Learning/Updating (Placeholder for future implementation)
    # - Get resulting_perception = get_perceptual_state_limited(sim)
    # - Store (current_perception, best_action, resulting_perception) in memory (e.g., a deque)
    # - If goal was reached, assign utilities to the trace and train/update the ANN utility model

if step_count >= MAX_STEPS:
    print(f"Reached maximum step count ({MAX_STEPS}). Stopping.")
    rob.stopMotors()

# --- Cleanup ---
print("Disconnecting...")
rob.disconnect()
sim.disconnect()