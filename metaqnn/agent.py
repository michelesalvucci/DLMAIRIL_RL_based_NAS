from typing import List
import random

from experience_replay import ExperienceReplay
from parameters import DEFAULT_Q_VALUE, INITIAL_R_SIZE, LAYERS_DEPTH_LIMIT, RL_AGENT_Q_LEARNING_RATE, SAMPLED_MODELS_FROM_EXPERIENCE_REPLAY
from models.layers import AbstractLayer, ConvolutionalLayer, PoolingLayer, FullyConnectedLayer, GlobalAveragePoolingLayer, SoftmaxLayer
from representation_size import calculate_new_representation_size, get_representation_size_bin


"""
Generate the stringified representation of a convolutional network configuration.
"""
class MetaQnnAgent:

    def __init__(self):
        """Fully connected layers selected so far"""
        self.fc_layers_selected = 0

        """Experience replay buffer for storing architecture experiences"""
        self.replay_memory = ExperienceReplay()
        self.update_counter = 0

        """Q-table: (state, action) -> Q-value"""
        # State: (object of type AbstractLayer with related parameters, representation_size_bin)
        # Action: (object of type AbstractLayer with related parameters)
        self.q_table = {}

        """Epsilon-greedy parameter"""
        self.epsilon = 1.0

    def generate_architecture(self) -> List[AbstractLayer]:
        """Generate complete CNN architecture using current policy"""
        architecture = []
        current_r_size = INITIAL_R_SIZE

        while True:
            action = self.__choose_action(architecture[-1] if architecture else None, current_r_size)
            architecture.append(action)
            
            # Update representation size
            current_r_size = calculate_new_representation_size(current_r_size, action)

            if action.type == 'SM' or action.type == 'GAP':
                break
        
        return architecture

    def update_with_experience_replay(self, architecture: List[AbstractLayer], reward: float):
        """
        Update Q-values using experience replay.
        Stores the architecture and reward in the replay buffer, then samples from it to update Q-values
        """
        self.replay_memory.add_experience(architecture, reward)

        self.__update_q_values(architecture, reward)

        """ After each model evaluation, sample 100 architectures from replay and update Q-values """
        if self.replay_memory.size() >= SAMPLED_MODELS_FROM_EXPERIENCE_REPLAY:
            batch = self.replay_memory.sample_batch(SAMPLED_MODELS_FROM_EXPERIENCE_REPLAY)
            for experience in batch:
                architecture = experience['architecture']
                reward = experience['reward']
                self.__update_q_values(architecture, reward)

    def __choose_action(self, state: AbstractLayer, current_r_size: float) -> AbstractLayer:
        """
        Choose an action based on the current state using epsilon-greedy policy.
        """
        valid_actions: List[AbstractLayer] = list(self.__get_valid_actions(state, current_r_size))
        state_key = self.__get_state_key(state, current_r_size)

        """Algorithm 2 - Appendix A of Baker et al. (2017, p. 12)"""
        if random.random() < self.epsilon or state_key not in self.q_table:
            """ Exploration: random action """
            return random.choice(valid_actions)
        else:
            """Exploitation: best action based on Q-values"""
            q_values = self.q_table[state_key]
            best_action = None
            best_q = float('-inf')

            for action in valid_actions:
                q_value = q_values.get(str(action), DEFAULT_Q_VALUE)
                if q_value > best_q:
                    best_q = q_value
                    best_action = action

        best_action = best_action if best_action else random.choice(valid_actions)
    
        if best_action.type == 'FC':
            self.fc_layers_selected += 1

        return best_action

    def __get_valid_actions(self, state: AbstractLayer, current_r_size: float):
        """
            Based on constraints defined in Baker et al. (2017)
            Including representation size constraints
        """

        next_layer_depth = state.layer_depth + 1 if state is not None else 0
        r_size_bin = get_representation_size_bin(current_r_size)

        """Based on Baker et al. (2017, p. 5)"""
        available_layers = ['C', 'P', 'FC', 'GAP', 'SM']

        if state is not None:
            if next_layer_depth == LAYERS_DEPTH_LIMIT:
                available_layers = ['SM'] if state.type == 'FC' else ['GAP', 'SM']
            else:
                if state.type == 'FC':
                    available_layers = ['FC', 'SM']
                if state.type == 'P' and 'P' in available_layers:
                    available_layers.remove('P')
                if (self.fc_layers_selected >= 2 or current_r_size > 8) and 'FC' in available_layers:
                    available_layers.remove('FC')

        """Parameters according to the table in  Baker et al. (2017, p. 5)"""
        """Convolutional (C)"""
        if 'C' in available_layers:
            for receptive_field_size in [1, 3, 5]:
                # Representation size constraint: receptive field size <= current r_size
                if receptive_field_size <= current_r_size:
                    for receptive_field in [64, 128, 256, 521]:
                        yield ConvolutionalLayer(
                            layer_depth=next_layer_depth,
                            receptive_fields=receptive_field,
                            field_size=receptive_field_size,
                            stride=1,  # Default stride
                        )

        """Pooling (P)"""
        if 'P' in available_layers:
            for receptive_field_size in [2, 3, 5]:
                # Representation size constraint: receptive field size <= current r_size
                if receptive_field_size <= current_r_size:
                    for stride in [2, 3]:
                        yield PoolingLayer(
                            layer_depth=next_layer_depth,
                            field_size=receptive_field_size,
                            stride=stride
                        )

        """Fully Connected (FC)"""
        """R-Size constraint according to Baker et al. (2017, p. 5)"""
        if 'FC' in available_layers:
            # FC layers are only allowed for medium and small representation sizes
            if r_size_bin in ['medium', 'small']:
                for neurons in [128, 256, 512]:
                    yield FullyConnectedLayer(
                        layer_depth=next_layer_depth,
                        neurons=neurons
                    )

        """Global Average Pooling (GAP)"""
        if 'GAP' in available_layers:
            yield GlobalAveragePoolingLayer(
                layer_depth=next_layer_depth
            )
        
        """Softmax (SM)"""
        if 'SM' in available_layers:
            yield SoftmaxLayer(
                layer_depth=next_layer_depth
            )

    def __update_q_values(self, architecture: List[AbstractLayer], reward: float):
        """
        Update the Q-value for each state-action pair using the Q-learning formula.
        Base on equation 3 in Baker et al. (2017, p. 3).
        Algorithm 3 - Appendix A of Baker et al. (2017, p. 12)
        """
        state = None
        current_r_size = INITIAL_R_SIZE
        
        for i, layer in enumerate(architecture):
            # State key including representation size
            state_key = self.__get_state_key(state, current_r_size) if i == 0 else self.__get_state_key(architecture[i-1], current_r_size)
            
            # Action taken
            action = str(layer)

            # Update representation size for next iteration
            current_r_size = calculate_new_representation_size(current_r_size, layer)

            # Next state (if not terminal)
            if i < len(architecture) - 1:
                next_state_key = self.__get_state_key(layer, current_r_size)
            else:
                next_state_key = None  # Terminal state

            # Update Q-value
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            Q_s1_a = self.q_table[state_key].get(action, DEFAULT_Q_VALUE)
            
            if next_state_key is None:
                max_Q_s2 = 0.0 # Terminal state
            else:
                if next_state_key in self.q_table:
                    max_Q_s2 = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else DEFAULT_Q_VALUE
                else:
                    max_Q_s2 = DEFAULT_Q_VALUE
            
            """ Q-learning update: the actual equation 3 in Baker et al. (2017, p. 3) """
            self.q_table[state_key][action] = (1 - RL_AGENT_Q_LEARNING_RATE) * Q_s1_a + RL_AGENT_Q_LEARNING_RATE * (reward + RL_AGENT_Q_LEARNING_RATE * max_Q_s2)


    def __get_state_key(self, layer: AbstractLayer, r_size: float) -> str:
        """Generate state key including representation size bin"""
        r_size_bin = get_representation_size_bin(r_size)
        if layer is None:
            return f"None__{r_size_bin}"
        return f"{str(layer)}__{r_size_bin}"
