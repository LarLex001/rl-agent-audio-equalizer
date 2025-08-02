import numpy as np


def actions_to_parameters(action_indices, action_space):
    return action_space.decode_action(action_indices)


class EQActionSpace:
    """
    Action space definition for audio equalizer control with 4-band EQ configuration.
    Defines discrete action spaces for frequency, gain, and Q-factor parameters
    across low shelf, low peak, high peak, and high shelf filter bands.
    
    Parameters:
        None (initialized with predefined frequency ranges, gain values, and Q-factor ranges)
        
    Attributes:
        freq_ranges: frequency ranges for each filter type (Hz)
        gain_values: gain adjustment values from -24.0 to 24.0 dB
        q_values: Q-factor values for shelf (0.1-2.0) and peak (0.1-32.0) filters
    """
    
    def __init__(self):

        self.freq_ranges = {
            'low_shelf': np.arange(20, 601, 20, dtype=int),        # 20-600 Hz
            'peak_low': np.arange(600, 6001, 100, dtype=int),      # 600-6000 Hz
            'peak_high': np.arange(6000, 12001, 200, dtype=int),   # 6000-12000 Hz
            'high_shelf': np.arange(12000, 20001, 500, dtype=int)  # 12000-20000 Hz
        }
        
        # gain values from -24.0 to 24.0 with 0.5 precision
        self.gain_values = np.round(np.arange(-24.0, 24.1, 0.5), 1)
        
        # Q values with different ranges for shelf and peak filters
        self.q_values = {
            'shelf': np.round(np.arange(0.1, 2.01, 0.1), 2),      # 0.10-2.00
            'peak': np.round(np.arange(0.1, 32.01, 0.1), 2)       # 0.10-32.00
        }

        self.freq_sizes = {k: len(v) for k, v in self.freq_ranges.items()}
        self.gain_size = len(self.gain_values)
        self.q_sizes = {k: len(v) for k, v in self.q_values.items()}
        
    def decode_action(self, action_indices):
        parameters = {}
        
        # decode freq
        for filter_type in self.freq_ranges.keys():
            freq_idx = action_indices[f'{filter_type}_freq']
            parameters[f'{filter_type}_freq'] = self.freq_ranges[filter_type][freq_idx]
        # decode gain
        for filter_type in self.freq_ranges.keys():
            gain_idx = action_indices[f'{filter_type}_gain']
            parameters[f'{filter_type}_gain'] = self.gain_values[gain_idx]
        # decode q
        for filter_type in self.freq_ranges.keys():
            q_idx = action_indices[f'{filter_type}_q']
            if 'shelf' in filter_type:
                parameters[f'{filter_type}_q'] = self.q_values['shelf'][q_idx]
            else:
                parameters[f'{filter_type}_q'] = self.q_values['peak'][q_idx]
        
        return parameters
    
    def get_action_space_size(self):
        action_space = {}
        
        for filter_type in self.freq_ranges.keys():
            action_space[f'{filter_type}_freq'] = self.freq_sizes[filter_type]
            action_space[f'{filter_type}_gain'] = self.gain_size
            
            if 'shelf' in filter_type:
                action_space[f'{filter_type}_q'] = self.q_sizes['shelf']
            else:
                action_space[f'{filter_type}_q'] = self.q_sizes['peak']
                
        return action_space
    