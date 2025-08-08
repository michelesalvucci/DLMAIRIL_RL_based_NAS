from typing import List
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from agent import MetaQnnAgent
from cnn_builder import build_cnn
from parameters import EPSILON_TO_NUM_MODELS_MAP, OUTPUT_DIR
from child_trainer import ChildTrainer
from models.layers import AbstractLayer


class NasResult:
    computed_weights: List[np.ndarray]
    validation_accuracy: float
    architecture: List[AbstractLayer]

    def __init__(self, computed_weights=None, validation_accuracy=0.0, architecture=None):
        self.computed_weights = computed_weights
        self.validation_accuracy = validation_accuracy
        self.architecture = architecture if architecture is not None else []


best_results: List[NasResult] = []


def run_neural_architecture_search():
    print("Running full Q-learning CNN architecture search...")
    
    trainer = ChildTrainer()
    agent = MetaQnnAgent()
    best_accuracy = 0.0
    
    """ Track accuracy history for plotting """
    accuracy_history = []
    best_accuracy_history = []
    episode_numbers = []
    episode = 0

    for epsilon, num_models in EPSILON_TO_NUM_MODELS_MAP.items():
        print(f"\n====================================================================================")
        print(f"\nExploration rate (epsilon): {epsilon:.4f} | Number of models to explore: {num_models}")
        print(f"\n====================================================================================")

        agent.epsilon = epsilon

        for _ in range(num_models):

            episode += 1
            architecture = agent.generate_architecture()
            model = build_cnn(architecture)

            try:
                print(f"\n--------------------------------------------------------")
                print(f"Evaluating episode {episode} with epsilon={epsilon}...")

                validation_accuracy, computed_weights = trainer.train_and_evaluate_architecture(model)

                """ Update Q-learning agent with experience replay """
                reward = validation_accuracy / 100.00
                agent.update_with_experience_replay(architecture, reward)
                
                # Track accuracy history
                accuracy_history.append(validation_accuracy)
                episode_numbers.append(episode)

                print(f"Episode {episode} - Validation accuracy: {validation_accuracy:.2f}% - Reward: {reward:.4f}")

                """ Track best architectures based on validation accuracy (keep at most 10, sorted) """
                if validation_accuracy > best_accuracy:
                    print(f"New best architecture found with validation accuracy: {validation_accuracy:.2f}%")
                    best_accuracy = validation_accuracy

                if validation_accuracy > best_accuracy or len(best_results) < 10:
                    new_result = NasResult(computed_weights, validation_accuracy, architecture)
                    best_results.append(new_result)
                    # Sort by validation_accuracy descending and keep only top 10
                    best_results.sort(key=lambda x: x.validation_accuracy, reverse=True)
                    if len(best_results) > 10:
                        best_results.pop()

                print(f"\n--------------------------------------------------------")
                
                # Update best accuracy history
                best_accuracy_history.append(best_accuracy)

                __log_to_csv(agent.epsilon, episode, reward, sum(w.size for w in computed_weights))

            except Exception as e:
                print(f"Error evaluating architecture: {e}")
    
    # Generate plot at the end
    __generate_final_plot(episode_numbers, accuracy_history, best_accuracy_history)

    __train_best_architectures(best_results, trainer)


def __generate_final_plot(episodes, accuracies, best_accuracies):
    """Generate the final accuracy plot with all data"""
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top plot: Accuracies
    ax1.plot(episodes, accuracies, 'b-', alpha=0.6, label='Current Accuracy')
    ax1.plot(episodes, best_accuracies, 'r-', linewidth=2, label='Best Accuracy')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Neural Architecture Search - Accuracy Progress')
    ax1.legend()
    ax1.grid(True)
    
    # Bottom plot: Epsilon values
    epsilon_values = []
    for epsilon, num_models in EPSILON_TO_NUM_MODELS_MAP.items():
        epsilon_values.extend([epsilon] * num_models)
    
    ax2.plot(episodes, epsilon_values[:len(episodes)], 'g-', linewidth=2, label='Epsilon')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon (Exploration Rate)')
    ax2.set_title('Exploration Rate Over Episodes')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot to outputs folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_filename = os.path.join(OUTPUT_DIR, "metaqnn_accuracy_progress.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.show()


def __train_best_architectures(best_results: List[NasResult], trainer: ChildTrainer):
    # Take best 10 architectures
    best_results.sort(key=lambda x: x.validation_accuracy, reverse=True)
    top_results = best_results[:10]

    results = []
    for i, result in enumerate(top_results):
        test_accuracy, weights = trainer.train_longer_and_evaluate_architecture(
            build_cnn(result.architecture), result.computed_weights
        )
        results.append({
            'architecture': result.architecture,
            'test_accuracy': test_accuracy,
            'weights': weights,
        })

    # Log the best 5 architectures after fine-tuning as a table
    results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    print("\nTop 5 architectures after fine-tuning:")
    print("{:<6} {:<20} {:<50} {:<10}".format("Rank", "Validation Accuracy (%)", "Architecture", "Parameters"))
    print("-" * 100)
    for i, arch in enumerate(results[:5]):
        arch_layers = " - ".join(f"{layer.type}{layer.short_description()}" for layer in arch['architecture'])
        # Compute total number of parameters from weights
        parameters = sum(w.size for w in arch.get('weights', []))
        print("{:<6} {:<20.2f} {:<50} {:<10}".format(i+1, arch['test_accuracy'], arch_layers, parameters))


def __log_to_csv(epsilon, episode, reward, parameters):
    # Log to CSV
    csv_filename = os.path.join(OUTPUT_DIR, "metaqnn_results.csv")
    # csv_filename = os.path.join(os.path.dirname(__file__), "outputs", "nas_results.csv")
    try:
        with open(csv_filename, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write header if file is empty
            if csv_file.tell() == 0:
                writer.writerow(['Epsilon', 'Episode', 'Reward', 'Parameters'])
            writer.writerow([epsilon, episode, reward, parameters])
    except Exception as csv_error:
        print(f"Error writing to CSV: {csv_error}")
