from rl_training import run_neural_architecture_search
import tensorflow as tf
import subprocess

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Physical devices:", tf.config.list_physical_devices())
print("NVIDIA-SMI check:", subprocess.getoutput(["nvidia-smi"]))


if __name__ == "__main__":
    run_neural_architecture_search()