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

    game = Game()
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
        game = Game()
        steps = 0
        max_steps = 500

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
            game.execute_placement(action)
            game.step()


            # Affichage
            if delay > 0 and (steps % 2 == 0):
                print(f"\nStep {steps + 1}: Action = {action}")
                visualize_game(game)

                time.sleep(delay)

            steps += 1

        # Résumé de l'épisode
        print(f"\n📊 Résumé Episode {ep + 1}:")
        print(f"  Score final: {game.score}")
        print(f"  Steps: {steps}")
        print(f"  Game over: {game.game_over}")

        if delay > 0 and ep < episodes - 1:
            input("\nAppuyez sur Entrée pour l'épisode suivant...")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_random_vs_trained()
    else:
        print("\n🎮 MODES DISPONIBLES :")
        print("  1. Visualisation lente (avec affichage)")
        print("  2. Visualisation rapide (résumés seulement)")
        print()

        mode = input("Choisissez un mode (1/2/3) : ").strip()

        if mode == "1":
            watch_agent_play(episodes=3, delay=0.3, load_model='save/tetris_simple_dqn_improved.pth')
        elif mode == "2":
            watch_agent_play(episodes=10, delay=0)
        else:
            print("Mode invalide")

