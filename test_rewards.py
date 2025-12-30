#!/usr/bin/env python3
"""
Test rapide pour vérifier que le système de reward fonctionne correctement.
"""

from game import Game
from agent import calculate_reward
import numpy as np

def test_reward_system():
    """Teste différents scénarios et affiche les rewards."""
    print("🧪 TEST DU SYSTÈME DE RÉCOMPENSE")
    print("=" * 70)

    # Test 1: Créer une grille, faire un DROP simple
    print("\n📋 Test 1: DROP simple sans ligne complétée")
    game = Game(12, 22)
    prev_state = game.get_state_features()

    # Simuler quelques actions
    actions_sequence = [4]  # DROP direct

    for action in actions_sequence:
        game.exec_action(action)
        game.step()
        reward = calculate_reward(game, prev_state, action)
        print(f"  Action: {action} (DROP), Reward: {reward:.2f}, Score: {game.score}, Game Over: {game.game_over}")
        prev_state = game.get_state_features()

        if game.game_over:
            break

    # Test 2: Jouer plusieurs pièces
    print("\n📋 Test 2: Plusieurs placements")
    game = Game(12, 22)
    prev_state = game.get_state_features()

    action_names = ["NONE", "LEFT", "RIGHT", "ROTATE", "DROP"]

    for i in range(10):  # 10 pièces
        print(f"\n  Pièce {i+1}:")
        steps = 0
        while not game.game_over and steps < 20:
            # Choisir une action aléatoire
            import random
            action = random.choice([1, 2, 3, 4])  # LEFT, RIGHT, ROTATE, DROP

            game.exec_action(action)
            game.step()
            reward = calculate_reward(game, prev_state, action)

            if abs(reward) > 0.5:  # Afficher seulement les rewards significatifs
                print(f"    {action_names[action]}: reward={reward:.2f}, score={game.score}")

            prev_state = game.get_state_features()
            steps += 1

            # Si DROP, on passe à la pièce suivante
            if action == 4:
                break

        if game.game_over:
            print(f"  ❌ Game Over après {i+1} pièces")
            break

    # Test 3: Analyser les rewards moyens
    print("\n📋 Test 3: Statistiques sur 100 actions aléatoires")
    game = Game(12, 22)
    prev_state = game.get_state_features()

    rewards = []
    reward_by_action = {0: [], 1: [], 2: [], 3: [], 4: []}

    import random
    for _ in range(100):
        if game.game_over:
            break

        action = random.choice([0, 1, 2, 3, 4])
        game.exec_action(action)
        game.step()
        reward = calculate_reward(game, prev_state, action)

        rewards.append(reward)
        reward_by_action[action].append(reward)
        prev_state = game.get_state_features()

    print(f"\n  Reward moyen global: {np.mean(rewards):.3f} (±{np.std(rewards):.3f})")
    print(f"  Reward min: {min(rewards):.2f}, max: {max(rewards):.2f}")
    print(f"\n  Par action:")
    for action, action_name in enumerate(action_names):
        if reward_by_action[action]:
            avg = np.mean(reward_by_action[action])
            print(f"    {action_name}: {avg:.3f} (n={len(reward_by_action[action])})")

    print("\n" + "=" * 70)
    print("✅ Tests terminés")

if __name__ == "__main__":
    test_reward_system()

