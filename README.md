# Reinforcement Learning for Neural Architecture Search: Experimental Research Project

This repository contains an experimental project developed for university research and training, focused on applying reinforcement learning methods to automate the design of convolutional neural network (CNN) architectures for image classification. The project includes two main approaches:

## Project Structure

- **nas_net_architecture**: Implements Neural Architecture Search (NAS) using the REINFORCE policy gradient algorithm, inspired by Zoph et al. (2017, 2018). The agent searches for optimal cell-based architectures, using a moving baseline to reduce variance and supporting both real and mock training for fast experimentation.
- **metaqnn**: Adapts the MetaQNN approach from Baker et al. (2017), using tabular Q-learning and experience replay to discover high-performing CNN architectures. The agent constructs architectures layer-by-layer, with strict constraints and an epsilon-greedy exploration schedule.

## Summary of Approaches

- Both modules use a custom "leaves" dataset (4,000 images, 64x64 pixels) for benchmarking.
- The NAS module supports cell-based architecture search and transfer learning, with mock training for rapid agent progression and optional real training for final evaluation.
- The MetaQNN module uses Q-learning with experience replay, strict layer constraints, and early stopping to efficiently explore the architecture space.
- After the search phase, the best architectures are retrained on the full dataset for final evaluation.

## Technical Specifications

- **Python 3.9** (main modules; some scripts may run with Python 3.11)
- **Virtual environment** (`venv`) created locally for dependency isolation
- **TensorFlow 2.12**
- **NumPy**
- **Matplotlib**
- **Pillow**
- **Gym**
- **Training performed on AWS SageMaker**, using **S3** for dataset and output storage

See the respective `requirements.txt` files in each module for full dependency details.

## Notes

- This is an experimental research project for university training purposes.
- The code is designed for flexibility and rapid experimentation, with support for both CPU and GPU environments.
- For further details, see the individual README files in `nas_net_architecture` and `metaqnn`.

## Bibliography

- Baker, B., Gupta, O., Naik, N., & Raskar, R. (2017). _Designing Neural Network Architectures using Reinforcement Learning_ (No. arXiv:1611.02167). arXiv. https://doi.org/10.48550/arXiv.1611.02167
- Zoph, B., & Le, Q. V. (2017). _Neural Architecture Search with Reinforcement Learning_ (No. arXiv:1611.01578). arXiv. https://doi.org/10.48550/arXiv.1611.01578
- Zoph, B., Vasudevan, V., Shlens, J., & Le, Q. V. (2018). _Learning Transferable Architectures for Scalable Image Recognition_ (No. arXiv:1707.07012). arXiv. https://doi.org/10.48550/arXiv.1707.07012