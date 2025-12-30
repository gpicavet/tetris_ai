#!/usr/bin/env python3
"""
Script de diagnostic pour comprendre pourquoi l'agent ne progresse pas.
Affiche des informations détaillées sur les récompenses, Q-values, etc.
"""

from agent import SimpleDQNAgent, calculate_reward
from game import Game
import numpy as np
import torch

def diagnose_agent(num_episodes=10):
    """Diagnostic approfondi de l'agent."""
    print("🔍 DIAGNOSTIC DE L'AGENT")
    print("=" * 70)

    game = Game(12, 22)
    state_size = len(game.get_state_features())
    action_size = 5

    agent = SimpleDQNAgent(
        state_size, action_size, grid_width=12,
        learning_rate=0.0005, gamma=0.99,
        epsilon=0.5,  # 50% exploration pour voir les deux comportements
        epsilon_decay=0.99, epsilon_min=0.01,
        batch_size=64
    )

    print(f"\n📊 Configuration :")
    print(f"  - State size: {state_size}")
    print(f"  - Action size: {action_size}")
    print(f"  - Device: {agent.device}")
    print(f"  - Hidden size: 256")
    print(f"  - Learning rate: 0.0005")
    print(f"  - Gamma: 0.99")

    all_rewards = []
    all_q_values = []
    state_features = []

    for ep in range(num_episodes):
        game = Game(12, 22)
        state = game.get_state_features()

        episode_rewards = []
        episode_q_values = []
        steps = 0
        max_steps = 100

        print(f"\n{'='*70}")
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"{'='*70}")

        while not game.game_over and steps < max_steps:
            actions = game.get_possible_actions()
            if not actions:
                break

            # Observer les Q-values
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q_vals = agent.q_network(state_tensor).squeeze(0).cpu().numpy()

            # Choisir action
            action_idx, action = agent.get_best_action(game, actions, training=True)
            if action is None:
                break

            # Exécuter
            game.exec_action(action)
            game.step()
            next_state = game.get_state_features()

            # Calculer récompense
            reward = calculate_reward(game, state, action)

            # Stocker
            episode_rewards.append(reward)
            episode_q_values.append(q_vals)

            # Affichage détaillé des premiers steps
            if steps < 5:
                print(f"\n  Step {steps + 1}:")
                print(f"    Action: {action}")
                print(f"    Q-values: [{', '.join(f'{q:.2f}' for q in q_vals)}]")
                print(f"    Reward: {reward:.3f}")
                print(f"    State features (selected):")
                print(f"      - Total height: {state[0]:.1f}")
                print(f"      - Total holes: {state[1]:.1f}")
                print(f"      - Bumpiness: {state[2]:.1f}")
                print(f"      - Empty columns: {state[5]:.1f}")

            # Mémoriser et apprendre
            agent.remember(state, action, reward, next_state, game.game_over)
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()

            state = next_state
            steps += 1

        # Stats de l'épisode
        all_rewards.extend(episode_rewards)
        all_q_values.extend(episode_q_values)
        state_features.append(state)

        avg_reward = np.mean(episode_rewards)
        avg_q = np.mean([q.max() for q in episode_q_values])

        print(f"\n  📈 Résumé :")
        print(f"    Score: {game.score}")
        print(f"    Steps: {steps}")
        print(f"    Reward moyen: {avg_reward:.3f}")
        print(f"    Reward total: {sum(episode_rewards):.2f}")
        print(f"    Q-value max moyen: {avg_q:.2f}")
        print(f"    Game over: {game.game_over}")

    # Analyse globale
    print(f"\n{'='*70}")
    print(f"📊 ANALYSE GLOBALE ({num_episodes} épisodes)")
    print(f"{'='*70}")

    all_rewards = np.array(all_rewards)
    print(f"\n🎁 Récompenses :")
    print(f"  - Moyenne: {all_rewards.mean():.3f}")
    print(f"  - Std: {all_rewards.std():.3f}")
    print(f"  - Min: {all_rewards.min():.3f}")
    print(f"  - Max: {all_rewards.max():.3f}")
    print(f"  - Médiane: {np.median(all_rewards):.3f}")

    # Distribution des récompenses
    print(f"\n  Distribution :")
    positive = (all_rewards > 0).sum()
    negative = (all_rewards < 0).sum()
    zero = (all_rewards == 0).sum()
    print(f"    Positives: {positive} ({100*positive/len(all_rewards):.1f}%)")
    print(f"    Négatives: {negative} ({100*negative/len(all_rewards):.1f}%)")
    print(f"    Nulles: {zero} ({100*zero/len(all_rewards):.1f}%)")

    # Q-values
    all_q_values = np.array(all_q_values)
    q_means = all_q_values.mean(axis=0)
    q_stds = all_q_values.std(axis=0)

    print(f"\n🎯 Q-values (moyenne par action) :")
    actions_names = ["NONE", "LEFT", "RIGHT", "ROTATE", "DROP"]
    for i, name in enumerate(actions_names):
        print(f"  {name:8s}: {q_means[i]:7.2f} ± {q_stds[i]:5.2f}")

    print(f"\n🧠 Réseau de neurones :")
    total_params = sum(p.numel() for p in agent.q_network.parameters())
    trainable_params = sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad)
    print(f"  - Paramètres totaux: {total_params:,}")
    print(f"  - Paramètres entraînables: {trainable_params:,}")
    print(f"  - Taille mémoire: {len(agent.memory)}")

    # Recommandations
    print(f"\n💡 RECOMMANDATIONS :")

    if all_rewards.mean() < -1:
        print("  ⚠️  Récompense moyenne très négative")
        print("      → Réduire les pénalités (hauteur, trous, bumpiness)")

    if all_rewards.std() < 1:
        print("  ⚠️  Faible variance des récompenses")
        print("      → L'agent a du mal à distinguer les bonnes/mauvaises actions")

    if abs(q_means.max() - q_means.min()) < 1:
        print("  ⚠️  Q-values très proches")
        print("      → Le réseau n'a pas encore appris de préférences")
        print("      → Continuer l'entraînement")

    if (all_q_values.max() > 100).any():
        print("  ⚠️  Q-values explosives")
        print("      → Réduire learning rate ou augmenter gradient clipping")

    print(f"\n✅ Si les Q-values sont différenciées et les récompenses variées,")
    print(f"   l'agent devrait progresser avec plus d'épisodes.")

if __name__ == "__main__":
    diagnose_agent(num_episodes=10)

