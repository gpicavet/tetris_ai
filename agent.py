from random import random, randrange, sample
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from game import Game

class SimpleQNetwork(nn.Module):
    """Réseau Q simple : état → Q-values pour toutes les actions."""
    def __init__(self, state_size, num_actions, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_actions)
        )
    
    def forward(self, state):
        """Retourne les Q-values pour toutes les actions."""
        return self.network(state)


class ReplayMemory:
    """Mémoire pour Experience Replay."""
    def __init__(self, capacity=50000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return sample(self.memory, min(batch_size, len(self)))
    
    def __len__(self):
        return len(self.memory)


class SimpleDQNAgent:
    def __init__(self, state_size, action_size, grid_width=12, learning_rate=0.0001, gamma=0.95, 
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1, batch_size=128):
        self.state_size = state_size
        self.grid_width = grid_width
        self.num_actions = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Un seul réseau
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                  'mps' if torch.backends.mps.is_available() else 'cpu')
        self.q_network = SimpleQNetwork(state_size, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayMemory()
        
        print(f"Device: {self.device}")
        print(f"Actions: {self.num_actions} (grid_width={grid_width})")
    
    def get_best_action(self, game, placements, training=True):
        """Choisit le meilleur placement."""
        if not placements:
            return None, None
        
        # Exploration aléatoire
        if training and random() < self.epsilon:
            idx = randrange(len(placements))
            return idx, placements[idx]
        
        # Exploitation : évaluer toutes les Q-values
        state = torch.FloatTensor(game.get_state_features()).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state).squeeze(0)  # (num_actions,)
        
        # Trouver le meilleur parmi les placements valides
        best_idx = 0
        best_q = float('-inf')
        
        for idx, placement in enumerate(placements):
            q = q_values[placement].item()
            if q > best_q:
                best_q = q
                best_idx = idx
        
        return best_idx, placements[best_idx]
    
    def remember(self, state, placement, reward, next_state, done):
        """Stocke une transition."""
        self.memory.push((state, placement, reward, next_state, done))
    
    def replay(self):
        """Entraîne le réseau sur un batch."""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Échantillonner
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convertir en tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q-values actuelles
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Q-values cibles
        with torch.no_grad():
            next_q = self.q_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename='simple_dqn.pth'):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename='simple_dqn.pth'):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


def calculate_reward(game):
    """Calcule la récompense."""
    if game.game_over:
        return -10
    
    reward = game.score * 50
    
    # Bonus survie
    reward += 1
    
    return reward


def train(episodes=3000, max_steps=500):
    """Entraîne l'agent."""
    game = Game(12, 22)
    state_size = len(game.get_state_features())
    action_size=5
    
    agent = SimpleDQNAgent(
        state_size,
        action_size,
        grid_width=12,
        learning_rate=0.0001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.1,
        batch_size=128
    )
    
    scores = []
    losses = []
    steps_per_episode = []
    
    for episode in range(episodes):
        game = Game(12, 22)
        state = game.get_state_features()
        episode_loss = []
        steps = 0
        
        while not game.game_over and steps < max_steps:
            # Obtenir placements possibles
            actions = game.get_possible_actions()
            if not actions:
                break

            # Choisir placement
            action_idx, action = agent.get_best_action(game, actions, training=True)
            if action is None:
                break
            
            # Exécuter
            game.exec_action(action)
            game.step()
            next_state = game.get_state_features()
            print(next_state)
            
            # Récompense
            reward = calculate_reward(game)
            
            # Mémoriser
            agent.remember(state, action, reward, next_state, game.game_over)
             
            # Suivant
            state = next_state
            steps += 1
        
        # Décroissance epsilon
        agent.decay_epsilon()
            
        # Apprendre
        loss = agent.replay()
        if loss > 0:
            episode_loss.append(loss)
       
        # Stats
        scores.append(game.score)
        steps_per_episode.append(steps)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
        
        # Affichage
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            avg_steps = np.mean(steps_per_episode[-50:])
            max_score = max(scores[-50:])
            
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Score: {avg_score:.2f} (max={max_score}) | "
                  f"Steps: {avg_steps:.1f} | "
                  f"Loss: {avg_loss:.3f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Mem: {len(agent.memory)}")
        
        # Sauvegarde
        if (episode + 1) % 500 == 0:
            agent.save(f'save/checkpoint_{episode+1}.pth')
    
    return agent, scores, losses, steps_per_episode


if __name__ == "__main__":
    
    # Entraîner
    agent, scores, losses, steps = train(episodes=3000)
    
    # Sauvegarder
    agent.save('save/tetris_simple_dqn.pth')
    print("\n✅ Modèle sauvegardé")
    
    # Graphiques
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Scores
        axes[0, 0].plot(scores, alpha=0.3)
        if len(scores) >= 100:
            axes[0, 0].plot(np.convolve(scores, np.ones(100)/100, mode='valid'), 
                           linewidth=2, label='Moy. mobile')
        axes[0, 0].set_title('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Steps
        axes[0, 1].plot(steps, alpha=0.3, color='green')
        if len(steps) >= 100:
            axes[0, 1].plot(np.convolve(steps, np.ones(100)/100, mode='valid'), 
                           linewidth=2, color='darkgreen')
        axes[0, 1].set_title('Steps par épisode')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss
        axes[1, 0].plot(losses, alpha=0.5, color='red')
        if len(losses) >= 50:
            axes[1, 0].plot(np.convolve(losses, np.ones(50)/50, mode='valid'), 
                           linewidth=2, color='darkred')
        axes[1, 0].set_title('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution
        recent = scores[-500:] if len(scores) >= 500 else scores
        axes[1, 1].hist(recent, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(np.mean(recent), color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Distribution scores (500 derniers)')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('results.png', dpi=150)
        print("📊 Graphiques sauvegardés")
    except ImportError:
        print("⚠️ matplotlib non installé")