# MetaQNN Reinforcement Learning for Neural Architecture Search: Adaptation and Implementation Notes

This project presents an adaptation of the MetaQNN approach introduced by Baker et al. (2017), which uses reinforcement learning—specifically $Q$-learning—to automate the design of convolutional neural network (CNN) architectures for image classification tasks.

## Project Motivation

The implementation was developed as part of a research assignment to explore and compare reinforcement learning-based neural architecture search (NAS) methods. The focus is on practical application and comparative analysis, rather than reproducing the original MetaQNN results at scale.

## Adaptation Highlights

- **Reduced Search Space and Training**: Fewer architectures were explored and each was trained for fewer epochs, with early stopping to save resources.
- **Simplified Experience Replay**: Experience replay was retained but with fewer samples.
- **Strict Layer Constraints**: Only valid CNN architectures were allowed, following the original paper's rules.
- **Adjusted Exploration**: The epsilon-greedy schedule was kept but with reduced values to lower computational demands.

## Implementation Details

- **Q-Learning**: The agent uses a tabular Q-learning approach, updating Q-values for state-action pairs after each episode, following the standard Bellman update.
- **State Representation**: Each state encodes the current layer, its parameters, and the representation size bin, enabling effective tracking of architecture construction.
- **Action Space**: At each step, the agent selects the next layer type and its parameters, constrained by the current state and architecture depth.
- **CNN Construction**: Architectures are translated into Keras models for training and evaluation.
- **Reward Signal**: Validation accuracy is used as the reward for each sampled architecture.

## Experimental Setup

- **Dataset**: A custom "leaves" dataset (4,000 images, 64x64 pixels) was used to benchmark the NAS process.
- **Training Environment**: All experiments were run on **AWS SageMaker GPU instances** (`ml.g5.xlarge`).
- **Evaluation**: The best architectures discovered were retrained and evaluated on a held-out test set.

## Results Summary

Despite the reduced search space and training time, the adapted MetaQNN agent was able to discover high-performing architectures, achieving up to 98.5% test accuracy on the custom dataset. The results demonstrate the effectiveness of reinforcement learning for NAS, even under significant resource constraints.

## Technologies

- **Python 3.9**
- **TensorFlow 2.12**
- **NumPy**
- **Matplotlib**
- **Pillow**
- **Gym**

All code is written in Python 3.9 and tested with the above libraries. See `requirements.txt` for details.

## References

- Baker, B., Gupta, O., Naik, N., & Raskar, R. (2017). Designing Neural Network Architectures using Reinforcement Learning. arXiv:1611.02167.

For further details, see the main code in the `metaqnn` folder.