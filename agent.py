from random import random, randrange, sample
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from game import Game

class SimpleQNetwork(nn.Module):
    """Réseau Q : état → Q-value unique pour évaluer la qualité d'un placement."""
    def __init__(self, state_size, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)  # UNE SEULE SORTIE : Q-value
        )
    
    def forward(self, state):
        """Retourne la Q-value pour un état donné."""
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
    def __init__(self, state_size, learning_rate=0.0005, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05, batch_size=64):
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Deux réseaux pour la stabilité (Target Network)
        self.device = torch.device('cpu')
        self.q_network = SimpleQNetwork(state_size, hidden_size=256).to(self.device)
        self.target_network = SimpleQNetwork(state_size, hidden_size=256).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(capacity=100000)

        self.update_target_every = 500  # Fréquence Mise à jour du target network
        self.steps = 0

        print(f"Device: {self.device}")

    def get_placement_state(self, game, placement):
        """
        Calcule l'état résultant après un placement donné (sans l'exécuter réellement).
        Retourne les features de l'état après le placement simulé.
        """
        import copy

        # Créer une copie du jeu
        sim_game = copy.deepcopy(game)

        # Exécuter le placement sur la copie
        x, rotation = placement

        # Appliquer la rotation
        for _ in range(rotation):
            sim_game.rotate()

        # Positionner en x
        while sim_game.piece_x < x:
            if not sim_game.translate(1):
                break
        while sim_game.piece_x > x:
            if not sim_game.translate(-1):
                break

        # Drop
        sim_game.drop()

        # Placer la pièce (mais ne pas générer de nouvelle pièce)
        sim_game._place_piece()
        sim_game._clear_lines()

        # Retourner les features de cet état
        return sim_game.get_state_features()

    def get_best_placement(self, game, placements, training=True):
        """Choisit le meilleur placement en évaluant tous les placements possibles."""
        if not placements:
            return None

        # Exploration aléatoire
        if training and random() < self.epsilon:
            idx = randrange(len(placements))
            return placements[idx]

        # Exploitation : évaluer tous les placements
        best_placement = None
        best_q = float('-inf')

        for placement in placements:
            # Obtenir l'état résultant après ce placement
            state = self.get_placement_state(game, placement)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Évaluer avec le réseau
            with torch.no_grad():
                q_value = self.q_network(state_tensor).item()

            if q_value > best_q:
                best_q = q_value
                best_placement = placement

        return best_placement

    def remember(self, placement_state, reward, next_state, done):
        """Stocke une transition.
        placement_state: état après le placement (features utilisées pour Q)
        reward: récompense obtenue
        next_state: état après step (avec nouvelle pièce)
        done: game over?
        """
        self.memory.push((placement_state, reward, next_state, done))

    def replay(self):
        """Entraîne le réseau sur un batch."""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Échantillonner
        batch = self.memory.sample(self.batch_size)
        placement_states, rewards, next_states, dones = zip(*batch)

        # Convertir en tensors
        placement_states = torch.FloatTensor(np.array(placement_states)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Q-values actuelles pour les placements effectués
        current_q = self.q_network(placement_states).squeeze(1)

        # Q-values cibles avec le target network
        # Pour next_state, on évalue directement (pas de max car 1 seule sortie)
        with torch.no_grad():
            next_q = self.target_network(next_states).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Mise à jour du target network
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename='simple_dqn.pth'):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename='simple_dqn.pth'):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']




def calculate_reward(game, next_state, prev_score):
    """
    Calcule la récompense après un placement complet.
    NOUVELLE APPROCHE : Le placement est déjà effectué, on évalue juste le résultat.
    """
    # 1. Pénalité game over (très forte)
    if game.game_over:
        return -10.0

    d_score = (game.score - prev_score)

    height = next_state[1]
    holes = next_state[2]
    bumpiness = next_state[3]


    return d_score - 0.1 * height - 0.3 * holes - 0.1 * bumpiness


def train(episodes=3000, max_steps=1000):
    """Entraîne l'agent avec la nouvelle architecture basée sur les placements."""
    game = Game()
    state_size = len(game.get_state_features())

    agent = SimpleDQNAgent(
        state_size,
        learning_rate=0.0005,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.1,
        batch_size=512
    )
    
    scores = []
    losses = []
    pieces_per_episode = []
    rewards_per_episode = []

    for episode in range(episodes):
        game = Game()
        episode_loss = []
        episode_rewards = []
        pieces = 0
        prev_score = 0

        while not game.game_over and pieces < max_steps:
            # Obtenir tous les placements possibles pour la pièce courante
            placements = game.get_possible_placements()
            if not placements:
                break

            # Choisir le meilleur placement
            placement = agent.get_best_placement(game, placements, training=True)
            if placement is None:
                break
            
            # Obtenir l'état après ce placement (pour la Q-value)
            placement_state = agent.get_placement_state(game, placement)

            # Exécuter le placement
            game.execute_placement(placement)
            pieces += 1

            # Obtenir le nouvel état (après placement et nouvelle pièce)
            next_state = game.get_state_features()

            # Calculer la récompense
            reward = calculate_reward(game, next_state, prev_score)
            episode_rewards.append(reward)
            prev_score = game.score

            # Mémoriser la transition
            agent.remember(placement_state, reward, next_state, game.game_over)

            # Apprendre à chaque placement
            loss = agent.replay()
            if loss > 0:
                episode_loss.append(loss)

        if pieces == max_steps:
            print(f"⚠️  Épisode {episode + 1} atteint le max de pièces ({max_steps})")

        # Décroissance epsilon
        agent.decay_epsilon()

        # Stats
        scores.append(game.score)
        pieces_per_episode.append(pieces)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
        total_reward = sum(episode_rewards)
        rewards_per_episode.append(total_reward)

        # Affichage
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            avg_pieces = np.mean(pieces_per_episode[-50:])
            max_score = max(scores[-50:])
            avg_reward = np.mean(rewards_per_episode[-50:])

            print(f"Episode {episode + 1}/{episodes} | "
                  f"Score: {avg_score:.2f} (max={max_score}) | "
                  f"Pièces: {avg_pieces:.1f} | "
                  f"Reward: {avg_reward:.1f} | "
                  f"Loss: {avg_loss:.3f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Mem: {len(agent.memory)}")
        
        # Sauvegarde
        if (episode + 1) % 500 == 0:
            agent.save(f'save/checkpoint_{episode+1}.pth')
    
    return agent, scores, losses, pieces_per_episode
