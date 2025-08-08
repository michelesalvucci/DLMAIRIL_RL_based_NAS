import tensorflow as tf
import matplotlib.pyplot as plt
from cnn_builder import build_child_network
from controller import SimpleNASController, decode_actions_to_cells
from child_trainer import ChildTrainer
from parameters import B, EPISODES, OUTPUT_DIR, SEARCH_DATASET_FRACTION
import numpy as np
import os
import csv


def run_nas_net():
    """
        The controller RNN generates 5 decisions per block (input 1, input 2, op 1, op 2, combine).
        This is repeated B times for each cell (where B is the number of blocks in a cell).
        The process is done for two types of cells: normal and reduction.
        Stages count: 2*B*5
    """
    controller = SimpleNASController()
    print("Simple REINFORCE controller initialized for NAS cell search.")
    
    cell_reward_map = {}
    reward_history = []
    trainer = ChildTrainer(SEARCH_DATASET_FRACTION)

    for episode in range(EPISODES):
        # Initialize state (all -1 means no actions taken yet)
        state = np.full(2 * B * 5, -1, dtype=np.float32)
        actions = []
        episode_states = []  # Store states for each step
        
        # Generate a complete architecture
        for stage in range(2 * B * 5):
            # Store current state before taking action
            current_state = state.copy()
            episode_states.append(current_state)
            
            action = controller.get_action(current_state, stage)
            actions.append(action)
            
            # Update state with the chosen action
            state[stage] = action
        
        # Mapping string representation of the architecture
        normal_cell, reduction_cell = decode_actions_to_cells(actions, B)
        cell_architecture_str = str(normal_cell) + " + " + str(reduction_cell)

        # Evaluate the complete architecture (mock or real training)
        model = build_child_network(normal_cell, reduction_cell, B)
        # reward = trainer.mock_train_and_evaluate_child_network(actions)
        reward, _ = trainer.train_and_evaluate_child_network(model)
        
        # Store all transitions with the final reward (REINFORCE principle)
        for stage in range(2 * B * 5):
            controller.store_transition(
                state=tf.constant(episode_states[stage], dtype=tf.float32),
                action=tf.constant(actions[stage], dtype=tf.int32),
                reward=tf.constant(reward, dtype=tf.float32)
            )
        
        # Train the controller
        baseline = controller.train_step()
        
        # Record results
        cell_reward_map[cell_architecture_str] = reward
        reward_history.append(reward)
        
        if baseline is not None:
            print(f"Episode {episode + 1}/{EPISODES} - Reward: {reward:.3f} - Baseline: {baseline:.3f}")
        else:
            print(f"Episode {episode + 1}/{EPISODES} - Reward: {reward:.3f}")
        
        # Print some statistics every 20 episodes
        if (episode + 1) % 20 == 0:
            recent_rewards = reward_history[-20:]
            print(f"Last 20 episodes - Mean: {np.mean(recent_rewards):.3f}, Max: {np.max(recent_rewards):.3f}, Min: {np.min(recent_rewards):.3f}")
            print(f"Overall - Mean: {np.mean(reward_history):.3f}, Max: {np.max(reward_history):.3f}")

    __plot_reward_progression(reward_history)
    __log_reward_history(reward_history)

    sorted_cells = sorted(cell_reward_map.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 architectures:")
    for i, (cell, reward) in enumerate(sorted_cells[:5]):
        print(f"{i+1}. Reward: {reward:.3f}")

    # train_best_architectures(sorted_cells)
    
    print("NAS cell search completed.")


def train_best_architectures(sorted_cells):
    """Train on full dataset the 10 best architectures"""
    trainer = ChildTrainer(1.0)
    best_architectures = [cell for cell, _ in sorted_cells[:10]]
    for i, cell in enumerate(best_architectures):
        print(f"\nTraining architecture {i+1}: {cell}")
        normal_cell_str, reduction_cell_str = cell.split(" + ")
        normal_cell = eval(normal_cell_str)
        reduction_cell = eval(reduction_cell_str)
        model = build_child_network(normal_cell, reduction_cell, B)
        final_accuracy, _ = trainer.train_and_evaluate_child_network(model)
        print(f"Final accuracy for architecture {i+1}: {final_accuracy:.2f}%")


def __log_reward_history(reward_history):
    csv_filename = os.path.join(OUTPUT_DIR, "nasnet_reward_history.csv")
    try:
        with open(csv_filename, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write header if file is empty
            if csv_file.tell() == 0:
                writer.writerow(['Episode', 'Reward'])
            for episode, reward in enumerate(reward_history, start=1):
                writer.writerow([episode, reward])
    except Exception as csv_error:
        print(f"Error writing to CSV: {csv_error}")


def __plot_reward_progression(reward_history):
    plt.figure(figsize=(8, 4))
    plt.plot(reward_history, linewidth=1)  # Thinner line, no marker
    plt.title("Cell Architecture Rewards During RL Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Accuracy)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "nas_net_reward_progression.png"))
    #plt.show()