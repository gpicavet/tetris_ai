import pygame
from pygame import Rect
from agent import SimpleDQNAgent

from game import Game

BLOCK_SIZE = 30
BLOCK_GAP = 2

Colors= [
     (0, 0, 0),
    (200, 218, 200),
     (83, 218, 63),
     (254, 251, 52),
    (1, 237, 250),
    (255, 200, 46),
    (221, 10, 178),
    (0, 119, 211)
]

def to_rect(pos: (int, int)):
    return Rect((pos[0]+1) * (BLOCK_SIZE + BLOCK_GAP), (pos[1]+1) * (BLOCK_SIZE + BLOCK_GAP), BLOCK_SIZE, BLOCK_SIZE)


def draw(game: Game):
    #walls
    for i in range(game.width + 2):
        pygame.Surface.fill(screen, "gray", to_rect([i-1, game.height-1]))
    for i in range(game.height):
        pygame.Surface.fill(screen, "gray", to_rect([-1, i-1]))
        pygame.Surface.fill(screen, "gray", to_rect([game.width, i-1]))
    #blocs
    for x in range(game.width):
        for y in range(game.height):
            if game.grid[y][x] > 0:
                pygame.Surface.fill(screen, 
                                    Colors[game.grid[y][x]], 
                                    to_rect([x, y-1]))
    #current piece
    for x in range(len(game.piece_data)):
        for y in range(len(game.piece_data)):
            if game.piece_data[y][x] > 0:
                pygame.Surface.fill(screen, 
                                    Colors[game.piece_data[y][x]], 
                                    to_rect([x+game.piece_x, y+game.piece_y-1]))




running = True

thegame = Game()

# pygame setup
pygame.init()
pygame.font.init() 
clock = pygame.time.Clock()
my_font = pygame.font.SysFont('Comic Sans MS', 30)
screen_w = (thegame.width + 2) * BLOCK_SIZE + (thegame.width + 1) * BLOCK_GAP
screen_h = (thegame.height + 1) * BLOCK_SIZE + (thegame.height) * BLOCK_GAP
screen = pygame.display.set_mode((screen_w,screen_h))

agent = SimpleDQNAgent(len(thegame.get_state_features()))
agent.load('save/tetris_simple_dqn.pth')

dt = 0
time = 0
steptime=0
key_time=0

while running:

    # every step time, chosse the best possible placement
    if not thegame.game_over and steptime > 0.2:
        
        actions = thegame.get_possible_placements()
        if not actions:
            break

        action = agent.get_best_placement(thegame, actions, training=False)
        if action is None:
            break

        thegame.execute_placement(action)

        steptime=0


    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        running = False

    if not thegame.game_over:
        if keys[pygame.K_UP] and key_time>0.3 :
            thegame.rotate()
            key_time=0
        if keys[pygame.K_DOWN]and key_time>0.3:
            thegame.drop()
            key_time=0
        if keys[pygame.K_LEFT]and key_time>0.1:
            thegame.translate(-1)
            key_time=0
        if keys[pygame.K_RIGHT]and key_time>0.1:
            thegame.translate(1)
            key_time=0

    #if not thegame.game_over and steptime > 0.3:
    #    thegame.step()
    #    steptime=0

    screen.fill("black")
    draw(thegame)
    
    if thegame.game_over:
        text_surface = my_font.render('GAME OVER', True, (255, 255, 255))
        text_width = text_surface.get_width()
        text_height = text_surface.get_height()
        screen.blit(text_surface, (screen_w//2-text_width//2,screen_h//2-text_height//2))

    pygame.display.flip()

    # limits FPS to 60
    dt = clock.tick(60) / 1000
    time += dt
    steptime += dt
    key_time += dt

pygame.quit()