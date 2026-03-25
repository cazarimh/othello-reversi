from __future__ import annotations
from game.utils import Player

class Knot():    
    def __init__(self, board: list[list[Player]], score: int, pos: tuple[int, int], depth: int):    
        self.board = board
        self.score = score
        self.pos = pos
        self.depth = depth
        self.children: list[Knot] = []

    def isLeaf(self):
        return len(self.children) == 0