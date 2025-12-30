from random import randrange

import numpy as np

COLORS = 6

TETROMINOS = [
    [
        [0,1,0,0],
        [0,1,0,0],
        [0,1,0,0],
        [0,1,0,0]
    ],
    [
        [1,1],
        [1,1],
    ],
    [
        [1,1,0],
        [0,1,1],
        [0,0,0],
    ],
    [
        [0,1,1],
        [1,1,0],
        [0,0,0],
    ],
    [
        [1,0,0],
        [1,1,1],
        [0,0,0]
    ],
    [
        [0,0,1],
        [1,1,1],
        [0,0,0]
    ],
    [
        [0,1,0],
        [1,1,1],
        [0,0,0],
    ]
]

class Placement:
    x: int
    r: int
    def __init__(self, x: int, r: int):
        self.x=x
        self.r=r

class Game:
    width: int
    height: int
    grid: list[list[int]]
    # piece courante
    piece_x: int
    piece_y: int
    piece_id: int
    piece_rot: int
    piece_data : list[list[int]]

    score: int
    game_over: bool

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[0] * width for _ in range(height)]
        self.score=0
        self.game_over=False
        self._new_piece()
    
    def step(self):

        if self._collides(self.piece_data, self.piece_x, self.piece_y+1):
            self._place_piece()
            
            self._clear_lines()

            self._new_piece()

        else:
            self.piece_y +=1
        

    def rotate(self):
        piece_size = len(self.piece_data)
        center_x = piece_size/2-0.5
        center_y = piece_size/2-0.5
        # rotation horaire 90° : Y=-X, X=Y dans le repere centré
        new_piece = [[self.piece_data[round(center_y-(x-center_x))][round(center_x+(y-center_y))] 
                      for x in range(piece_size)] for y in range(piece_size)]

        if not self._collides(new_piece, self.piece_x, self.piece_y):
            self.piece_data = new_piece
            self.piece_rot = (self.piece_rot+1) % 4
            return True
        
        return False

    def translate(self, dx:int):
        if not self._collides(self.piece_data, self.piece_x+dx, self.piece_y):
            self.piece_x += dx
            return True
        
        return False

    def drop(self):
        while not self._collides(self.piece_data, self.piece_x, self.piece_y+1):
            self.piece_y+=1

    def _place_piece(self):
        for y in range(len(self.piece_data)):
            for x in range(len(self.piece_data[y])):
                if self.piece_data[y][x]>0:
                    self.grid[self.piece_y+y][self.piece_x+x] = self.piece_data[y][x]

    def _new_piece(self):
        self.piece_id = randrange(len(TETROMINOS))
        color = 1+randrange(COLORS)
        piece = TETROMINOS[self.piece_id]
        piece_size = len(piece)
        self.piece_data = [[piece[y][x]*color for x in range(piece_size)] for y in range(piece_size)]
        self.piece_x = round(self.width /2 - len(self.piece_data)/2)
        self.piece_y = 0
        self.piece_rot = 0

        # game over
        if self._collides(self.piece_data, self.piece_x, self.piece_y):
            self.game_over = True
            
        return self.game_over

    def _collides(self,piece : list[list[int]], offset_x:int, offset_y:int) -> bool:    
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] >0:
                    # avec les bords
                    if offset_x+x<0 or offset_x+x>=self.width or offset_y+y>=self.height:
                        return True
                    # collision avec les autres blocs
                    if self.grid[offset_y+y][offset_x+x]>0:
                        return True
        return False

    
    def _clear_lines(self):
        lines=0
        y=self.height-1
        while y>=0:
            #a t-on une ligne ?
            is_full = all(self.grid[y][x] > 0 for x in range(self.width))
            if is_full:
                #decale toute les lignes supérieures d'un cran vers le bas
                for y2 in range(y, 1, -1):
                    self.grid[y2] = self.grid[y2-1][:]
                lines +=1
            else:
                y -=1

        self.score += lines * lines

        return lines
    
    def get_state_features(self) -> list:
        # Calcul des hauteurs
        heights = []
        for x in range(self.width):
            height = 0
            for y in range(self.height):
                if self.grid[y][x] > 0:
                    height = self.height - y
                    break
            heights.append(height)
        
        # Trous par colonne
        holes = []
        for x in range(self.width):
            col_holes = 0
            block_found = False
            for y in range(self.height):
                if self.grid[y][x] > 0:
                    block_found = True
                elif block_found:
                    col_holes += 1
            holes.append(col_holes)
        
        # FEATURES AGRÉGÉES
        features = [
            # 1. Hauteurs
            max(heights) / self.height,                    # Max height normalisé
            sum(heights) / (len(heights) * self.height),   # Avg height normalisé
            np.std(heights) / self.height,                 # Écart-type hauteurs
            
            # 2. Trous
            sum(holes),                                    # Total trous
            max(holes) if holes else 0,                    # Max trous par colonne
            
            # 3. Bumpiness (irrégularité)
            sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1)) / self.height,
            
            # 4. Puits (colonnes très basses entourées de hautes)
            sum(1 for i in range(1, len(heights)-1) 
                if heights[i] < heights[i-1] - 2 and heights[i] < heights[i+1] - 2),
            
            # 5. Lignes presque complètes (potentiel de scoring)
            sum(1 for y in range(self.height)
                if sum(1 for x in range(self.width) if self.grid[y][x] > 0) >= self.width - 2),
            
            # 6. Colonnes vides
            sum(1 for h in heights if h == 0),
            
            # 7. Type de pièce (one-hot ou ID)
            self.piece_id / 6.0  # Normalisé
        ]
    
        return np.array(features, dtype=np.float32)


    def get_possible_actions(self):
        return (0,1,2,3,4)
    
    def exec_action(self, action:int):

        res=True
        if action==0:
            pass
        elif action==1:
            res=self.translate(-1)
        elif action==2:
            res=self.translate(1)
        elif action==3:
            res=self.rotate()
        elif action==4:
            res=self.drop()

        return res
