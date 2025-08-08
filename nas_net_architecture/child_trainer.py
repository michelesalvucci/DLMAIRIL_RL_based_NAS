import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import BinaryCrossentropy
from parameters import CHILD_NETWORK_BATCH_SIZE, CHILD_NETWORK_EPOCHS, CHILD_NETWORK_LEARNING_RATE, CHILD_NETWORK_LEARNING_RATE_DECAY, CHILD_NETWORK_LEARNING_RATE_MOMENTUM, DATASET_DIR, IMG_SIZE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class ChildTrainer:
    train_data = None
    val_data = None
    test_data = None

    def __init__(self, fraction=1.0):
        self.load_data(fraction=fraction)

    def load_data(self, fraction=1.0):
        def get_label_from_filename(filename):
            return 1 if filename.startswith('LEAF_') else 0

        images = []
        labels = []

        print("Loading dataset from:", DATASET_DIR)
        
        for fname in os.listdir(DATASET_DIR):
            img_path = os.path.join(DATASET_DIR, fname)
            if os.path.isfile(img_path):
                img = load_img(img_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(get_label_from_filename(fname))

        print(f"Loaded {len(images)} images from {DATASET_DIR}")

        X = np.array(images, dtype='float32') / 255.0
        y = np.array(labels).astype('float32')  # Ensure labels are float for sigmoid

        # Shuffle the data before splitting
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        # Take only a fraction of the data if specified
        if fraction < 1.0:
            n_samples = int(fraction * len(X))
            X = X[:n_samples]
            y = y[:n_samples]
            print(f"Using only {n_samples} samples ({fraction*100:.1f}%) for this load.")

        # 30% for test, 70% for training/validation
        split_idx = int(0.7 * len(X))
        X_trainval, X_test = X[:split_idx], X[split_idx:]
        y_trainval, y_test = y[:split_idx], y[split_idx:]

        # Further split training/validation into train and validation (e.g., 80/20 split)
        val_split_idx = int(0.89 * len(X_trainval)) # Zoph and Le selected 11% of training data for validation
        X_train, X_val = X_trainval[:val_split_idx], X_trainval[val_split_idx:]
        y_train, y_val = y_trainval[:val_split_idx], y_trainval[val_split_idx:]

        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)

        print(f"train_data shape: {self.train_data[0].shape}, {self.train_data[1].shape}")
        print(f"val_data shape: {self.val_data[0].shape}, {self.val_data[1].shape}")
        print(f"test_data shape: {self.test_data[0].shape}, {self.test_data[1].shape}")

    def mock_train_and_evaluate_child_network(self, actions):
        """
        Create a wider reward range that correctly interprets different action types.
        """
        actions = np.array(actions)
        
        # Separate different types of actions
        operation_actions = []
        input_actions = []
        combine_actions = []
        
        # Extract actions by type (every 5 actions = 1 block)
        for i in range(0, len(actions), 5):
            if i + 4 < len(actions):  # Ensure we have a complete block
                input_actions.extend([actions[i], actions[i+1]])  # input1, input2
                operation_actions.extend([actions[i+2], actions[i+3]])  # op1, op2
                combine_actions.append(actions[i+4])  # combine
        
        # Analyze operation choices (these are the ones that should be 0-12)
        operation_actions = np.array(operation_actions)
        high_op_count = sum(op > 8 for op in operation_actions) if len(operation_actions) > 0 else 0
        low_op_count = sum(op < 4 for op in operation_actions) if len(operation_actions) > 0 else 0
        
        # Analyze input diversity (variety in connections)
        input_diversity = len(set(input_actions)) / max(1, len(set(range(max(input_actions) + 1)))) if input_actions else 0
        
        # Analyze combination strategy
        combine_variety = len(set(combine_actions)) / max(1, len(combine_actions)) if combine_actions else 0
        
        total_ops = len(operation_actions)
        if total_ops == 0:
            return 0.3  # Fallback if no operations
            
        high_op_ratio = high_op_count / total_ops
        low_op_ratio = low_op_count / total_ops
        
        # More sophisticated reward function
        reward_base = 0.5  # Higher base to start at 0.5
        
        # Operation-based rewards (main component) - make these larger
        operation_bonus = high_op_ratio * 0.4  # Bonus for high operations
        operation_penalty = low_op_ratio * 0.1  # Smaller penalty to avoid going too low
        
        # Architecture complexity rewards
        input_bonus = input_diversity * 0.15
        combine_bonus = combine_variety * 0.1
        
        # Operation diversity bonus
        op_diversity = len(set(operation_actions)) / 13.0  # How many different ops used
        diversity_bonus = op_diversity * 0.15
        
        # Non-linear components for more variation
        op_mean = np.mean(operation_actions) if len(operation_actions) > 0 else 6.5
        nonlinear_bonus = 0.13 * np.sin(op_mean * np.pi / 6.5)
        
        final_reward = (reward_base + 
                       operation_bonus - 
                       operation_penalty + 
                       input_bonus + 
                       combine_bonus + 
                       diversity_bonus + 
                       nonlinear_bonus)
        
        # Interpolate final reward to be more diverse
        final_reward *= final_reward
        
        # Ensure bounds are 0.5 to 0.98
        final_reward = max(0.5, min(0.998, final_reward))

        return final_reward


    def train_and_evaluate_child_network(self, model: Sequential) -> tuple:
        """ Shuffle training data before each training session """
        indices = np.arange(len(self.train_data[0]))
        np.random.shuffle(indices)
        X_train_shuffled = self.train_data[0][indices]
        y_train_shuffled = self.train_data[1][indices]
        self.train_data = (X_train_shuffled, y_train_shuffled)

        """ Compile model with optimizer and loss function """
        cross_entropy = BinaryCrossentropy(from_logits=False)
        # Use learning rate schedule instead of deprecated 'decay' parameter
        lr_schedule = ExponentialDecay(
            initial_learning_rate=CHILD_NETWORK_LEARNING_RATE,
            decay_steps=1000,
            decay_rate=CHILD_NETWORK_LEARNING_RATE_DECAY,
            staircase=True
        )
        optimizer = SGD(learning_rate=lr_schedule, momentum=CHILD_NETWORK_LEARNING_RATE_MOMENTUM)
        model.compile(optimizer=optimizer, loss=cross_entropy, metrics=['accuracy'])

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        history = model.fit(
            x=self.train_data[0],
            y=self.train_data[1],
            epochs=CHILD_NETWORK_EPOCHS,
            batch_size=CHILD_NETWORK_BATCH_SIZE,
            validation_data=(self.val_data[0], self.val_data[1]),
            callbacks=[early_stopping],
            verbose=0
        )

        _, test_accuracy = model.evaluate(
            x=self.test_data[0],
            y=self.test_data[1],
            batch_size=128,
            verbose=1
        )

        # The reward for each architecture is the cube of the maximum validation accuracy from the last five epochs as in Zoph and Le (2017)
        val_accuracies = history.history.get('val_accuracy', [])
        if len(val_accuracies) >= 5:
            max_val_acc = max(val_accuracies[-5:])
        else:
            max_val_acc = max(val_accuracies) if val_accuracies else 0.0
        reward = max_val_acc ** 3
        
        return reward, model.get_weights()
