#!/usr/bin/env python3
"""Script d'entraînement amélioré pour l'agent Tetris DQN."""

from agent import train
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("🎮 Démarrage de l'entraînement avec PLACEMENTS...")
    print("=" * 60)
    print("📌 Nouvelle architecture:")
    print("  - Le réseau évalue la qualité d'un PLACEMENT COMPLET")
    print("  - 1 seule sortie Q-value (pas 5 actions)")
    print("  - L'agent choisit parmi tous les placements possibles")
    print("=" * 60)

    # Entraîner avec les nouveaux paramètres
    agent, scores, losses, pieces = train(episodes=3000, max_steps=500)

    # Sauvegarder le modèle final
    agent.save('save/tetris_simple_dqn_improved.pth')
    print("\n✅ Modèle amélioré sauvegardé")

    # Graphiques détaillés
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Scores
        axes[0, 0].plot(scores, alpha=0.3, label='Score brut')
        if len(scores) >= 100:
            moving_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
            axes[0, 0].plot(range(99, len(scores)), moving_avg,
                           linewidth=2, label='Moy. mobile 100', color='red')
        axes[0, 0].set_title('Score par épisode')
        axes[0, 0].set_xlabel('Épisode')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Steps
        axes[0, 1].plot(pieces, alpha=0.3, color='green', label='Pièces brut')
        if len(pieces) >= 100:
            moving_avg_pieces = np.convolve(pieces, np.ones(100)/100, mode='valid')
            axes[0, 1].plot(range(99, len(pieces)), moving_avg_pieces,
                           linewidth=2, color='darkgreen', label='Moy. mobile 100')
        axes[0, 1].set_title('Pièces placées par épisode')
        axes[0, 1].set_xlabel('Épisode')
        axes[0, 1].set_ylabel('Pièces')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Loss
        axes[1, 0].plot(losses, alpha=0.5, color='red', label='Loss')
        if len(losses) >= 50:
            moving_avg_loss = np.convolve(losses, np.ones(50)/50, mode='valid')
            axes[1, 0].plot(range(49, len(losses)), moving_avg_loss,
                           linewidth=2, color='darkred', label='Moy. mobile 50')
        axes[1, 0].set_title('Loss d\'entraînement')
        axes[1, 0].set_xlabel('Épisode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')  # Échelle log pour mieux voir

        # Distribution des scores (500 derniers)
        recent = scores[-500:] if len(scores) >= 500 else scores
        axes[1, 1].hist(recent, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        mean_recent = np.mean(recent)
        axes[1, 1].axvline(mean_recent, color='red', linestyle='--',
                          linewidth=2, label=f'Moyenne: {mean_recent:.2f}')
        axes[1, 1].set_title(f'Distribution scores ({len(recent)} derniers épisodes)')
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_ylabel('Fréquence')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('results_improved.png', dpi=150)
        print("📊 Graphiques sauvegardés dans 'results_improved.png'")

        # Statistiques finales
        print("\n" + "=" * 60)
        print("📈 STATISTIQUES FINALES")
        print("=" * 60)
        print(f"Score moyen (500 derniers) : {np.mean(scores[-500:]):.2f}")
        print(f"Score maximum : {max(scores)}")
        print(f"Pièces moyennes (500 derniers) : {np.mean(pieces[-500:]):.1f}")
        print(f"Loss finale : {losses[-1]:.3f}")
        print(f"Epsilon final : {agent.epsilon:.3f}")
        print(f"Taille mémoire : {len(agent.memory)}")

    except ImportError:
        print("⚠️ matplotlib non installé - impossible de générer les graphiques")

