# test_game.py - À exécuter séparément
from game import Game
from random import randrange

print("🔍 Diagnostic du jeu Tetris\n")

# Test 1: Initialisation
game = Game(12, 22)
print(f"1. Initialisation")
print(f"   Grille: {game.width}×{game.height}")
print(f"   Score initial: {game.score}")
print(f"   Game over: {game.game_over}")
print()

# Test 2: État initial
state = game.get_state_features()
print(f"2. État initial")
print(f"   Features: {state}")
print()

# Test 3: Placements possibles
placements = game.get_all_possible_placements()
print(f"3. Placements possibles")
print(f"   Nombre: {len(placements)}")
if placements:
    print(f"   Premier: {placements[0]}")
    print(f"   Type: {type(placements[0])}")
    
    # Test 4: Appliquer un placement
    placement, new_game = placements[0]

    # Continuer quelques placements
    print()
    print(f"5. Simulation de 10 placements")
    current_game = new_game
    for i in range(10):
        state = current_game.get_state_features()
        print(f"   Features: {state}")

        placements = current_game.get_all_possible_placements()
        if not placements:
            print(f"   Step {i}: AUCUN placement possible!")
            break
        if placements[0][1].game_over:
            print(f"   Step {i}: Game over!")
            break
        
        placement, current_game = placements[randrange(len(placements))]
        print(f"   Step {i}: Score={current_game.score}, placement={placement.x},  Placements={len(placements)}")
else:
    print("   ⚠️ AUCUN placement possible dès le départ!")