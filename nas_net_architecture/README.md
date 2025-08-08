# NAS with Reinforcement Learning: Adaptation and Implementation Notes

This project presents an adaptation of the Neural Architecture Search (NAS) methods introduced by Zoph et al. (2017, 2018), which use reinforcement learning—specifically the REINFORCE algorithm—to automate the design of convolutional neural network (CNN) architectures for image classification.

## Project Motivation

The implementation was developed as part of a research assignment to explore and compare reinforcement learning-based NAS methods. The focus is on practical application and comparative analysis, rather than reproducing the original NAS results at scale.

## Adaptation Highlights

- **REINFORCE with Baseline**: The controller is trained using the REINFORCE policy gradient algorithm, with a moving baseline to reduce variance, as described in Zoph et al. (2017).
- **Cell-based Search Space**: The search space is defined in terms of "cells" (normal and reduction), following Zoph et al. (2018), enabling transferability and modularity.
- **Mock Training for Fast Experimentation**: Due to high computational costs, a mock reward function is provided to simulate training and enable rapid agent progression, even on CPU-only machines.
- **Real Dataset Support**: The framework supports real model training on leaves dataset, but this is optional due to resource constraints.
- **Fine-tuning Best Architectures**: After the search, the top 10 architectures are retrained on the full dataset for final evaluation.

## Implementation Details

- **Controller**: A neural network samples architecture decisions (actions) for each cell block. Actions include input selection, operation type, and combination method.
- **Baseline**: A moving average of recent rewards is used as a baseline to compute the advantage for policy updates.
- **Cell Encoding**: Architectures are encoded as sequences of actions, which are decoded into normal and reduction cell structures.
- **Training**: The `ChildTrainer` class handles data loading, splitting, and (optionally) real model training and evaluation.
- **Mock Reward**: The `mock_train_and_evaluate_child_network()` function provides a non-linear reward based on architectural properties, allowing fast RL agent feedback.

## Experimental Setup

- **Dataset**: The "leaves" dataset (4,000 images, 64x64 pixels) is used for benchmarking.
- **Training Environment**: Experiments can be run on local machines (CPU) or cloud GPU instances.
- **Search Phase**: Only a fraction (e.g., 20%) of the dataset is used during the search to accelerate experimentation.
- **Evaluation Phase**: The best architectures are retrained on the full dataset for final accuracy measurement.

## Technologies

- **Python 3.9**
- **TensorFlow 2.12**
- **NumPy**
- **Matplotlib**

All code is written in Python 3.9 and tested with the above libraries. See `requirements.txt` for details.

## References

- Zoph, B., & Le, Q. V. (2017). Neural Architecture Search with Reinforcement Learning. arXiv:1611.01578.
- Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018). Learning Transferable Architectures for Scalable Image Recognition. arXiv:1707.07012.

For further details, see the main code in the `nas_net_architecture` folder.