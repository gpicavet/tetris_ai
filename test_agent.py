import torch
from agent import DQNAgent
from game import Game

def test_agent(agent, num_games=10, render=False):
    """Teste l'agent entraîné."""
    scores = []
    
    for game_num in range(num_games):
        game = Game(12,22)
        state = game.get_state_features()
        steps = 0
        
        while not game.game_over and steps < 1000:
            action_idx = agent.get_action(state, training=False)
            action = agent.actions[action_idx]
            
            if action == 'left':
                game.translate(-1)
            elif action == 'right':
                game.translate(1)
            elif action == 'rotate':
                game.rotate()
            elif action == 'drop':
                game.drop()
            
            game.step()
            state = game.get_state_features()
            steps += 1
        
        scores.append(game.score)
        print(f"Game {game_num + 1}: Score={game.score}, Steps={steps}")
    
    print(f"\n📊 Statistiques:")
    print(f"Score moyen: {np.mean(scores):.1f} (±{np.std(scores):.1f})")
    print(f"Meilleur score: {max(scores)}")

if __name__ == "__main__":
    print("🧠 Entraînement DQN pour Tetris\n")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")

    agent = DQNAgent(12+1+1+1+1, 4, learning_rate=0.0005)
    agent.load('dqn_checkpoint_1000.pth')
    
    # Test
    print("\n🧪 Test de l'agent:")
    test_agent(agent, num_games=20)