from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping
import os
import numpy as np
import gc

from parameters import BATCH_SIZE, DATASET_DIR, EARLY_STOPPING_PATIENCE, EPOCHS, IMG_SIZE, LEARNING_RATE


class ChildTrainer:
    train_data = None
    test_data = None

    def __init__(self):
        self.load_data()

    def load_data(self):
        def get_label_from_filename(filename):
            return 1 if filename.startswith('LEAF_') else 0

        images = []
        labels = []

        print("Loading dataset from:", DATASET_DIR)
        
        for fname in os.listdir(DATASET_DIR):
            img_path = os.path.join(DATASET_DIR, fname)
            if os.path.isfile(img_path):
                img = load_img(img_path, color_mode='grayscale', target_size=IMG_SIZE)
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

        # 30% for test, 70% for training/validation
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.train_data = (X_train, y_train)
        self.test_data = (X_test, y_test)
        self.validation_data = self.__load_validation_data()

        print(f"train_data shape: {self.train_data[0].shape}, {self.train_data[1].shape}")
        print(f"test_data shape: {self.test_data[0].shape}, {self.test_data[1].shape}")

    def __load_validation_data(self):
        X, y = self.train_data
        val_size = int(0.2 * len(X))
        
        # Create proper train/validation split
        X_train_only = X[val_size:]
        y_train_only = y[val_size:]
        X_val = X[:val_size]
        y_val = y[:val_size]
        
        # Update train_data to exclude validation data
        self.train_data = (X_train_only, y_train_only)
        
        return X_val, y_val

    def train_and_evaluate_architecture(self, model: Sequential) -> tuple:
        """ Shuffle training data before each training session """
        indices = np.arange(len(self.train_data[0]))
        np.random.shuffle(indices)
        X_train_shuffled = self.train_data[0][indices]
        y_train_shuffled = self.train_data[1][indices]
        self.train_data = (X_train_shuffled, y_train_shuffled)

        """ Compile model with optimizer and loss function """
        cross_entropy = BinaryCrossentropy(from_logits=False)
        optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.5)

        model.compile(optimizer=optimizer, loss=cross_entropy, metrics=['accuracy'])
        
        """ Create adaptive learning rate callback """
        adaptive_lr_callback = AdaptiveLearningRateCallback(initial_lr=LEARNING_RATE)

        """ Interrupt training if validation accuracy does not improve to save time """
        early_stop = EarlyStopping(monitor='val_accuracy', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True)

        model.fit(
            x=self.train_data[0],
            y=self.train_data[1],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=self.validation_data,
            callbacks=[adaptive_lr_callback, early_stop],
            verbose=0
        )
        
        # Log if early stopping occurred
        if early_stop.stopped_epoch > 0:
            print(f"Early stopping triggered at epoch {early_stop.stopped_epoch + 1}")
        else:
            print("Training completed without early stopping")
        
        # Evaluate accuracy on validation data
        _, val_accuracy = model.evaluate(
            x=self.validation_data[0],
            y=self.validation_data[1],
            batch_size=BATCH_SIZE,
            verbose=1
        )

        weights = model.get_weights()

        K.clear_session()
        del model
        gc.collect()
        
        return val_accuracy * 100.00, weights


    def train_longer_and_evaluate_architecture(self, model: Sequential, initial_weights=None):
        """
            Train best CNN architecture for more epochs and evaluate on test set.
            Optionally initialize training with previously found weights.
        """

        """ Shuffle training data before each training session """
        indices = np.arange(len(self.train_data[0]))
        np.random.shuffle(indices)
        X_train_shuffled = self.train_data[0][indices]
        y_train_shuffled = self.train_data[1][indices]
        self.train_data = (X_train_shuffled, y_train_shuffled)

        """ Compile model with optimizer and loss function and saved weights from validation stage"""
        if initial_weights is not None:
            model.set_weights(initial_weights)

        cross_entropy = BinaryCrossentropy(from_logits=False)
        optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=0.5)

        model.compile(optimizer=optimizer, loss=cross_entropy, metrics=['accuracy'])
        
        # Create adaptive learning rate callback for longer training
        adaptive_lr_callback = AdaptiveLearningRateCallback(initial_lr=LEARNING_RATE)
        
        model.fit(
            x=self.train_data[0],
            y=self.train_data[1],
            epochs=EPOCHS * 2,  # Train for double the epochs
            batch_size=BATCH_SIZE,
            validation_data=None,  # No validation data, only train accuracy
            callbacks=[adaptive_lr_callback],
            verbose=1
        )

        _, test_accuracy = model.evaluate(
            x=self.test_data[0],
            y=self.test_data[1],
            batch_size=BATCH_SIZE,
            verbose=1
        )

        K.clear_session()

        return test_accuracy * 100, model.get_weights()  # Convert to percentage


"""
    If the model failed to perform better than a random predictor after the first epoch, 
    we reduced the learning rate by a factor of 0.4 and restarted training, 
    for a maximum of 5 restarts. For models that started learning 
    (i.e., performed better than a random predictor), 
    we reduced the learning rate by a factor of 0.2 every 5 epochs
    (Baker et al., 2017, p. 6).
"""
class AdaptiveLearningRateCallback(Callback):
    def __init__(self, initial_lr=LEARNING_RATE):
        super().__init__()
        self.initial_lr = initial_lr
        self.random_predictor_threshold = 0.5  # 50% for binary classification
        self.max_restarts = 5

        self.restart_count = 0
        self.learning_started = False
        
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        
        # Check after first epoch if model is learning
        if epoch == 0:
            if val_acc <= self.random_predictor_threshold:
                # Model failed to perform better than random predictor
                if self.restart_count < self.max_restarts:
                    self.restart_count += 1
                    new_lr = self.initial_lr * (0.4 ** self.restart_count)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    print(f"\nRestart {self.restart_count}: Reducing learning rate to {new_lr}")
                    # Reset model weights for restart
                    for layer in self.model.layers:
                        if hasattr(layer, 'kernel_initializer'):
                            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
                        if hasattr(layer, 'bias_initializer') and layer.bias is not None:
                            layer.bias.assign(layer.bias_initializer(layer.bias.shape))
                else:
                    print("Maximum restarts reached")
            else:
                self.learning_started = True
                
        # Reduce learning rate every 5 epochs if learning started
        elif self.learning_started and (epoch + 1) % 5 == 0:
            current_lr = float(self.model.optimizer.learning_rate)
            new_lr = current_lr * 0.2
            self.model.optimizer.learning_rate.assign(new_lr)
            print(f"\nEpoch {epoch + 1}: Reducing learning rate to {new_lr}")
