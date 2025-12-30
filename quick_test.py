#!/usr/bin/env python3
"""Script de test rapide pour vérifier les améliorations."""

from agent import SimpleDQNAgent, calculate_reward
from game import Game
import numpy as np

def quick_test(episodes=100, max_steps=500):
    """Test rapide sur quelques épisodes."""
    game = Game(12, 22)
    state_size = len(game.get_state_features())
    action_size = 5

    agent = SimpleDQNAgent(
        state_size,
        action_size,
        grid_width=12,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9997,
        epsilon_min=0.01,
        batch_size=64
    )

    scores = []
    steps_list = []
    losses = []

    print("🔥 TEST RAPIDE - 100 ÉPISODES")
    print("=" * 60)

    for episode in range(episodes):
        game = Game(12, 22)
        state = game.get_state_features()
        episode_loss = []
        steps = 0

        while not game.game_over and steps < max_steps:
            actions = game.get_possible_actions()
            if not actions:
                break

            action_idx, action = agent.get_best_action(game, actions, training=True)
            if action is None:
                break

            game.exec_action(action)
            game.step()
            next_state = game.get_state_features()

            reward = calculate_reward(game, state, action)
            agent.remember(state, action, reward, next_state, game.game_over)

            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                if loss > 0:
                    episode_loss.append(loss)

            state = next_state
            steps += 1

        agent.decay_epsilon()

        scores.append(game.score)
        steps_list.append(steps)
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)

        if (episode + 1) % 10 == 0:
            recent_scores = scores[-10:]
            recent_steps = steps_list[-10:]
            print(f"Ep {episode + 1:3d} | "
                  f"Score: {np.mean(recent_scores):.2f} (max={max(recent_scores)}) | "
                  f"Steps: {np.mean(recent_steps):.1f} | "
                  f"Loss: {avg_loss:.3f} | "
                  f"ε: {agent.epsilon:.3f}")

    print("\n" + "=" * 60)
    print("📊 RÉSULTATS")
    print("=" * 60)
    print(f"Score moyen (50 derniers) : {np.mean(scores[-50:]):.2f}")
    print(f"Score maximum : {max(scores)}")
    print(f"Steps moyen (50 derniers) : {np.mean(steps_list[-50:]):.1f}")
    print(f"Loss finale : {losses[-1]:.3f}")
    print(f"Epsilon final : {agent.epsilon:.3f}")
    print(f"Mémoire : {len(agent.memory)} transitions")

    # Analyse des récompenses
    print("\n📈 ANALYSE DES RÉCOMPENSES")
    print(f"Nombre d'épisodes avec score > 0 : {sum(1 for s in scores if s > 0)}")
    print(f"Nombre d'épisodes avec score > 1 : {sum(1 for s in scores if s > 1)}")

    return agent, scores, steps_list, losses

if __name__ == "__main__":
    agent, scores, steps, losses = quick_test(episodes=100)

    # Afficher les 10 meilleurs scores
    top_10 = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:10]
    print("\n🏆 TOP 10 SCORES")
    for i, (ep, score) in enumerate(top_10, 1):
        print(f"  {i}. Episode {ep+1}: {score} points")

