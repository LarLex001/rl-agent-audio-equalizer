import torch.nn as nn


class DQNModel(nn.Module):
    """
    This model uses a deep feedforward neural network with modern regularization techniques
    to predict optimal Q-values for different equalizer actions based on audio features.
    
    Parameters:
        input_size: number of input features (audio feature vector size)
        output_size: number of possible actions (EQ parameter combinations)
        hidden_size: size of hidden layers (default: 512)
        dropout_rate: dropout probability for regularization (default: 0.2)
    """

    def __init__(self, input_size, output_size, hidden_size=512, dropout_rate=0.2):
        super(DQNModel, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size // 4, output_size)
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    