#!/usr/bin/env python3
"""
Visualisation en temps réel de l'agent jouant à Tetris.
Utile pour comprendre le comportement de l'agent.
"""

import time
from agent import SimpleDQNAgent, calculate_reward
from game import Game

def visualize_game(game):
    """Affiche l'état du jeu dans le terminal."""
    print("\n" + "=" * (game.width * 2 + 2))
    for row in game.grid:
        line = "|"
        for cell in row:
            line += "██" if cell > 0 else "  "
        line += "|"
        print(line)
    print("=" * (game.width * 2 + 2))
    print(f"Score: {game.score} | Pièce: {game.piece_id} | Pos: ({game.piece_x}, {game.piece_y})")

def watch_agent_play(episodes=5, load_model=None, delay=0.1):
    """Regarde l'agent jouer en temps réel."""
    print("👀 VISUALISATION DE L'AGENT")
    print("=" * 70)

    game = Game(12, 22)
    state_size = len(game.get_state_features())

    agent = SimpleDQNAgent(
        state_size
    )

    if load_model:
        try:
            agent.load(load_model)
            print(f"✅ Modèle chargé : {load_model}")
            print(f"   Epsilon: {agent.epsilon}")
        except FileNotFoundError:
            print(f"⚠️  Modèle non trouvé : {load_model}")
            print(f"   Utilisation d'un agent non entraîné")
    else:
        print("⚠️  Aucun modèle chargé - agent aléatoire")

    for ep in range(episodes):
        game = Game(12, 22)
        state = game.get_state_features()
        steps = 0
        max_steps = 500
        total_reward = 0

        print(f"\n{'='*70}")
        print(f"🎮 Episode {ep + 1}/{episodes}")
        print(f"{'='*70}")

        if delay > 0:
            visualize_game(game)
            time.sleep(delay * 2)

        while not game.game_over and steps < max_steps:
            actions = game.get_possible_placements()
            if not actions:
                break

            # Choisir action
            action = agent.get_best_placement(game, actions, training=False)
            if action is None:
                break

            # Exécuter
            prev_score = game.score
            game.execute_placement(action)
            game.step()
            next_state = game.get_state_features()

            # Récompense
            reward = calculate_reward(game, game.score)
            total_reward += reward

            # Affichage
            if delay > 0 and (steps % 2 == 0):
                print(f"\nStep {steps + 1}: Action = {action}")
                visualize_game(game)

                # Info sur la récompense si significative
                if reward > 1 or reward < -1:
                    print(f"💰 Reward: {reward:.2f}")
                if game.score > prev_score:
                    print(f"🎉 LIGNE(S) COMPLÉTÉE(S) ! Score +{game.score - prev_score}")

                time.sleep(delay)

            state = next_state
            steps += 1

        # Résumé de l'épisode
        print(f"\n📊 Résumé Episode {ep + 1}:")
        print(f"  Score final: {game.score}")
        print(f"  Steps: {steps}")
        print(f"  Reward total: {total_reward:.2f}")
        print(f"  Reward moyen/step: {total_reward/steps:.3f}")
        print(f"  Game over: {game.game_over}")

        if delay > 0 and ep < episodes - 1:
            input("\nAppuyez sur Entrée pour l'épisode suivant...")

def compare_random_vs_trained():
    """Compare un agent aléatoire vs entraîné."""
    print("🆚 COMPARAISON : Aléatoire vs Entraîné")
    print("=" * 70)

    results = {"random": [], "trained": []}

    # Agent aléatoire
    print("\n🎲 Agent aléatoire (10 épisodes)...")
    game = Game(12, 22)
    state_size = len(game.get_state_features())

    agent_random = SimpleDQNAgent(
        state_size, 5, grid_width=12,
        epsilon=1.0,  # 100% aléatoire
        epsilon_decay=1.0
    )

    for _ in range(10):
        game = Game(12, 22)
        state = game.get_state_features()
        steps = 0

        while not game.game_over and steps < 500:
            actions = game.get_possible_actions()
            if not actions:
                break

            action_idx, action = agent_random.get_best_action(game, actions, training=True)
            if action is None:
                break

            game.exec_action(action)
            game.step()
            state = game.get_state_features()
            steps += 1

        results["random"].append((game.score, steps))

    # Agent entraîné (si disponible)
    print("\n🧠 Agent entraîné (10 épisodes)...")
    try:
        agent_trained = SimpleDQNAgent(
            state_size, 5, grid_width=12,
            epsilon=0.0  # 0% aléatoire
        )
        agent_trained.load('save/tetris_simple_dqn.pth')

        for _ in range(10):
            game = Game(12, 22)
            state = game.get_state_features()
            steps = 0

            while not game.game_over and steps < 500:
                actions = game.get_possible_actions()
                if not actions:
                    break

                action_idx, action = agent_trained.get_best_action(game, actions, training=False)
                if action is None:
                    break

                game.exec_action(action)
                game.step()
                state = game.get_state_features()
                steps += 1

            results["trained"].append((game.score, steps))
    except FileNotFoundError:
        print("⚠️  Aucun modèle entraîné trouvé (save/tetris_simple_dqn.pth)")
        return

    # Comparaison
    print(f"\n{'='*70}")
    print("📊 RÉSULTATS")
    print(f"{'='*70}")

    import numpy as np

    random_scores = [s for s, _ in results["random"]]
    random_steps = [st for _, st in results["random"]]
    trained_scores = [s for s, _ in results["trained"]]
    trained_steps = [st for _, st in results["trained"]]

    print(f"\n🎲 Agent Aléatoire :")
    print(f"  Score moyen: {np.mean(random_scores):.2f} (max={max(random_scores)})")
    print(f"  Steps moyen: {np.mean(random_steps):.1f}")

    print(f"\n🧠 Agent Entraîné :")
    print(f"  Score moyen: {np.mean(trained_scores):.2f} (max={max(trained_scores)})")
    print(f"  Steps moyen: {np.mean(trained_steps):.1f}")

    print(f"\n📈 Amélioration :")
    score_improvement = (np.mean(trained_scores) - np.mean(random_scores)) / max(np.mean(random_scores), 0.01) * 100
    steps_improvement = (np.mean(trained_steps) - np.mean(random_steps)) / max(np.mean(random_steps), 1) * 100
    print(f"  Score: {score_improvement:+.1f}%")
    print(f"  Survie: {steps_improvement:+.1f}%")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_random_vs_trained()
    else:
        print("\n🎮 MODES DISPONIBLES :")
        print("  1. Visualisation lente (avec affichage)")
        print("  2. Visualisation rapide (résumés seulement)")
        print("  3. Comparaison aléatoire vs entraîné")
        print()

        mode = input("Choisissez un mode (1/2/3) : ").strip()

        if mode == "1":
            watch_agent_play(episodes=3, delay=0.3, load_model='save/tetris_simple_dqn_improved.pth')
        elif mode == "2":
            watch_agent_play(episodes=10, delay=0)
        elif mode == "3":
            compare_random_vs_trained()
        else:
            print("Mode invalide")

