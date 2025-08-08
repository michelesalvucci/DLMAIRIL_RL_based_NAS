import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras import layers, Model
from parameters import HIDDEN_STATE_OPERATIONS, POLICY_NETWORK_LEARNING_RATE, B, POLICY_NETWORK_UNITS, POLICY_TEMPERATURE_DECAY


class SimpleNASController:
    def __init__(self):
        self.num_actions = 2 * B * 5 # Total decisions needed (50 for B=5)
        
        # Different action spaces for different decision types (According to Zoph et al. 2018)
        self.max_inputs = 10  # Maximum number of possible input connections
        self.num_operations = len(HIDDEN_STATE_OPERATIONS)  # 13 operations
        self.num_combine_ops = 2  # addition or concatenation
        self.action_space_size = max(self.max_inputs, self.num_operations, self.num_combine_ops)
        
        # Policy network
        self.policy_net = self._build_policy_network(POLICY_NETWORK_UNITS)
        self.optimizer = Adam(learning_rate=POLICY_NETWORK_LEARNING_RATE)
        
        # Storage for episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Reward tracking for baseline
        self.reward_history = []
        self.baseline_window = 100
        self.best_reward_seen = 0.5
        
        # Temperature annealing parameters
        self.temperature = 5.0
        self.temperature_min = 1.0
        self.temperature_decay = POLICY_TEMPERATURE_DECAY
        self.episode_count = 0

    def get_action(self, state, stage):
        """Get action using current policy, constrained by decision type"""
        # Pass the current state to the policy network, returns action probabilities for this state (at most action_space_size)
        action_probs = self.policy_net(tf.expand_dims(state, 0))
        
        # Determine what type of decision this stage is
        decision_type = self._get_decision_type(stage)
        
        # Determine valid action space based on decision type
        if decision_type == 'input':
            block_idx = (stage % (2 * B * 5)) // 5  # Which block we're in
            available_actions = min(2 + block_idx, self.max_inputs)
            action_probs = action_probs[0, :available_actions]
        elif decision_type == 'operation':
            available_actions = self.num_operations
            action_probs = action_probs[0, :available_actions]
        elif decision_type == 'combine':
            available_actions = self.num_combine_ops
            action_probs = action_probs[0, :available_actions]

        # Convert "back" probabilities to logits before applying temperature scaling, since softmax uses exp(logits)
        logits = tf.math.log(action_probs + 1e-8) / self.temperature
        # And then "back" to probabilities according
        action_probs = tf.nn.softmax(logits)
        action_probs = action_probs / tf.reduce_sum(action_probs) # Just for numerical stability

        # Action selection based on probabilities of the softmax distribution
        # Tensorflow expects the logits to be in a 2D shape for categorical sampling
        action = tf.random.categorical(tf.math.log([action_probs + 1e-8]), 1)[0, 0]
        action = int(action.numpy())
        
        return action

    def store_transition(self, state, action, reward):
        """Store transition data"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def train_step(self):
        """Perform one training step using REINFORCE"""            
        # Convert to tensors
        states = tf.stack(self.episode_states)
        actions = tf.stack(self.episode_actions)
        rewards = tf.stack(self.episode_rewards)
        
        # Get the final reward (only the last step has non-zero reward)
        final_reward = rewards[-1]
        self.reward_history.append(float(final_reward))
        
        # Update best reward seen
        if final_reward > self.best_reward_seen:
            self.best_reward_seen = float(final_reward)
        
        # Calculate moving baseline from recent episodes
        if len(self.reward_history) > self.baseline_window:
            self.reward_history = self.reward_history[-self.baseline_window:]

        if len(self.reward_history) >= 20:
            recent_mean = np.mean(self.reward_history[-20:])  # Recent performance
            historical_mean = np.mean(self.reward_history)    # Overall performance
            best_baseline = self.best_reward_seen * 0.8       # 80% of best ever seen
            
            # Use the highest of these to prevent baseline from dropping too much
            baseline = max(recent_mean, historical_mean, best_baseline)
        else:
            baseline = 0.65
        
        # Difference = reward - baseline (positive for good episodes)
        difference = final_reward - baseline
        
        # Only train if we have a reasonable signal (avoid tiny difference)
        if abs(difference) < 0.05:
            # Skip training on episodes that are too close to baseline
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
            # Anneal temperature even if skipping training
            self._anneal_temperature()
            return baseline

        differences = tf.fill(tf.shape(rewards), difference)
        
        with tf.GradientTape() as tape:
            # Get action probabilities for all states
            action_probs = self.policy_net(states)
            
            # Get probabilities of taken actions
            action_indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            selected_action_probs = tf.gather_nd(action_probs, action_indices)
            
            # REINFORCE loss: -log(prob) * difference
            # Policy gradient update: theta = theta + alpha * E[pi_theta](s, a) * (R_k - b)
            # Add 1e-8 for numerical stability in log()
            # tf.reduce_mean computes the expected value
            # 'difference' represents (R_k - b), i.e., reward minus baseline
            loss = -tf.reduce_mean(tf.math.log(selected_action_probs + 1e-8) * differences)
        
        # Compute and apply gradients of the loss with respect to parameters, with clipping to avoid exploding gradients
        gradients = tape.gradient(loss, self.policy_net.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 0.3) for grad in gradients if grad is not None]
        if clipped_gradients:
            self.optimizer.apply_gradients(zip(clipped_gradients, self.policy_net.trainable_variables))
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

        # Anneal temperature after each episode
        self._anneal_temperature()

        return baseline

    def _build_policy_network(self, hidden_units):
        inputs = layers.Input(shape=(self.num_actions,), name='state')
        x = layers.Dense(hidden_units, activation='relu')(inputs)
        x = layers.Dense(hidden_units, activation='relu')(x)
        # Output layer: probability distribution over maximum action space
        outputs = layers.Dense(self.action_space_size, activation='softmax')(x)
        return Model(inputs=inputs, outputs=outputs, name='policy_network')
    
    def _get_decision_type(self, step):
        """Determine what type of decision this step represents"""
        step_in_block = step % 5
        if step_in_block in [0, 1]:  # input1, input2
            return 'input'
        elif step_in_block in [2, 3]:  # op1, op2
            return 'operation'
        else:  # step_in_block == 4, combine
            return 'combine'
    
    def _anneal_temperature(self):
        """Anneal the softmax temperature after each episode."""
        self.episode_count += 1
        self.temperature = max(
            self.temperature_min,
            self.temperature * self.temperature_decay
        )


def decode_actions_to_cells(actions, B):
    """Decode action sequence into cell structures"""
    def decode_cell(action_slice):
        cell_structure = []
        for block in range(B):
            idx = block * 5
            input1 = action_slice[idx] % (2 + block)  # Constrain to available inputs
            input2 = action_slice[idx+1] % (2 + block)
            op1 = HIDDEN_STATE_OPERATIONS[action_slice[idx+2] % len(HIDDEN_STATE_OPERATIONS)]
            op2 = HIDDEN_STATE_OPERATIONS[action_slice[idx+3] % len(HIDDEN_STATE_OPERATIONS)]
            combine = "element-wise-addition" if action_slice[idx+4] % 2 == 0 else "concatenation"
            block_pred = {
                "input1": input1,
                "input2": input2,
                "op1": op1,
                "op2": op2,
                "combine": combine,
                "output": f"block_{block}_output"
            }
            cell_structure.append(block_pred)
        return cell_structure

    actions = np.array(actions)
    normal_cell_action = actions[:B*5]
    reduction_cell_action = actions[B*5:2*B*5]
    normal_cell_structure = decode_cell(normal_cell_action)
    reduction_cell_structure = decode_cell(reduction_cell_action)
    return normal_cell_structure, reduction_cell_structure
