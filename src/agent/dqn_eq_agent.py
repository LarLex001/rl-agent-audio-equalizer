import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from src.agent.dqn_model import DQNModel
from src.agent.prioritized_replay_buffer import PrioritizedReplayBuffer


class EQAgent:
    """
    DQN agent for audio equalizer control with prioritized experience replay,
    double DQN, and separate neural networks for each EQ parameter.
    
    Parameters:
        feature_size: size of input audio feature vector
        action_space: dictionary mapping EQ parameters to their action space sizes
        device: computing device ('cuda' or 'cpu', default: 'cuda')
        hidden_size: size of hidden layers in neural networks (default: 512)
    """

    def __init__(self, feature_size, action_space, device='cuda', hidden_size=512):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.action_space = action_space
        self.hidden_size = hidden_size

        self.models = {}
        self.target_models = {}
        self.optimizers = {}
        self.schedulers = {}
        
        for param, size in action_space.items():
            self.models[param] = DQNModel(feature_size, size, hidden_size).to(self.device)
            self.target_models[param] = DQNModel(feature_size, size, hidden_size).to(self.device)
            self.target_models[param].load_state_dict(self.models[param].state_dict())

            self.optimizers[param] = optim.AdamW(
                self.models[param].parameters(), 
                lr=0.0005,  
                weight_decay=1e-4
            )
            self.schedulers[param] = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizers[param], 
                mode='min', 
                factor=0.8, 
                patience=10
            )

        self.memory = PrioritizedReplayBuffer(capacity=20000, alpha=0.6)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995 
        self.epsilon_min = 0.05      
        self.batch_size = 128       
        self.beta = 0.4           
        self.beta_increment = 0.001

        self.update_target_counter = 0
        self.target_update_freq = 100  
        self.soft_update_tau = 0.005  

        self.grad_clip = 1.0

        self.reward_history = []
        self.loss_history = {}
        self.epsilon_history = []
        
        for param in action_space.keys():
            self.loss_history[param] = []

    def select_action(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            actions = {}
            for param, size in self.action_space.items():
                if np.random.rand() < 0.7:  
                    actions[param] = np.random.randint(0, size)
                else:  
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        q_values = self.models[param](state_tensor)
                        best_action = q_values.argmax().item()

                        noise = np.random.randint(-min(5, size//10), min(5, size//10) + 1)
                        actions[param] = np.clip(best_action + noise, 0, size - 1)
        else:       
            actions = {}
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                for param, model in self.models.items():
                    q_values = model(state_tensor)
                    actions[param] = q_values.argmax().item()
        
        return actions

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return {}

        batch, indices, weights = self.memory.sample(self.batch_size, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        losses = {}
        all_td_errors = []

        states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*batch)
        
        for param, model in self.models.items():
            states = torch.tensor(np.array(states_list), dtype=torch.float32).to(self.device)
            actions = torch.tensor([a[param] for a in actions_list], dtype=torch.long).to(self.device)
            rewards = torch.tensor(rewards_list, dtype=torch.float32).to(self.device)
            next_states = torch.tensor(np.array(next_states_list), dtype=torch.float32).to(self.device)
            dones = torch.tensor(dones_list, dtype=torch.float32).to(self.device)
            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
            
            # Double DQN
            with torch.no_grad():
                next_actions = model(next_states).argmax(1)
                next_q_values = self.target_models[param](next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target_q = rewards + self.gamma * next_q_values * (1 - dones)
            
            current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # TD error
            td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
            all_td_errors.extend(td_errors)
            
            # Weighted loss
            loss = (weights_tensor * F.mse_loss(current_q, target_q, reduction='none')).mean()
            
            # Optimize with gradient clipping
            self.optimizers[param].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            self.optimizers[param].step()
            
            # Update learning rate
            self.schedulers[param].step(loss.item())
            
            losses[param] = loss.item()
            self.loss_history[param].append(loss.item())

        new_priorities = np.abs(all_td_errors[:len(indices)]) + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        # Soft update target networks
        self.update_target_counter += 1
        if self.update_target_counter % self.target_update_freq == 0:
            for param in self.models.keys():
                self._soft_update_target_network(param)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.epsilon_history.append(self.epsilon)
        
        return losses

    def _soft_update_target_network(self, param):
        for target_param, local_param in zip(
            self.target_models[param].parameters(), 
            self.models[param].parameters()
        ):
            target_param.data.copy_(
                self.soft_update_tau * local_param.data + 
                (1.0 - self.soft_update_tau) * target_param.data
            )

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        for param, model in self.models.items():
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizers[param].state_dict(),
                'scheduler_state_dict': self.schedulers[param].state_dict(),
                'epsilon': self.epsilon,
                'beta': self.beta,
                'hidden_size': self.hidden_size
            }
            torch.save(checkpoint, os.path.join(path, f'{param}_model.pt'))
    
    def load_model(self, path):
        for param, model in self.models.items():
            model_path = os.path.join(path, f'{param}_model.pt')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizers[param].load_state_dict(checkpoint['optimizer_state_dict'])
                self.schedulers[param].load_state_dict(checkpoint['scheduler_state_dict'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.beta = checkpoint.get('beta', self.beta)
                self.target_models[param].load_state_dict(model.state_dict())

    def plot_progress(self):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))

        # Reward history
        if self.reward_history:
            axes[0].plot(self.reward_history)
            axes[0].set_title('Reward History')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Reward')
            axes[0].grid(True)

        # Loss history
        for param, loss_history in self.loss_history.items():
            if loss_history:
                axes[1].plot(loss_history, label=param, alpha=0.7)
        
        axes[1].set_title('Loss History')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # Epsilon history
        if self.epsilon_history:
            axes[2].plot(self.epsilon_history)
            axes[2].set_title('Epsilon Decay')
            axes[2].set_xlabel('Training Step')
            axes[2].set_ylabel('Epsilon')
            axes[2].grid(True)
        
        plt.tight_layout()
        return fig
    