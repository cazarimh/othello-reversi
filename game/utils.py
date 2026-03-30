from enum import IntEnum, Enum
import numpy as np

class Player(IntEnum):
	EMPTY = np.int8(0)
	BLACK = np.int8(1)
	WHITE = np.int8(2)

class BoardHouses(Enum):
	CORNER = np.int8(10)
	X = np.int8(-5)
	C = np.int8(-3)
	A = np.int8(-2)
	B = np.int8(-1)
	SIMPLE = np.int8(1)
	MIDDLE = np.int8(2)

class Direction(Enum):
	N = (-1, 0)
	NE = (-1, 1)
	E = (0, 1)
	SE = (1, 1)
	S = (1, 0)
	SW = (1, -1)
	W = (0, -1)
	NW = (-1, -1)

class Directions():
	def getAllDirections():
		return [
			Direction.N,
			Direction.NE,
			Direction.E,
			Direction.SE,
			Direction.S,
			Direction.SW,
			Direction.W,
			Direction.NW
		]

	def oppositeDirection(direction: Direction) -> Direction:
		return Direction((direction.value[0]*-1, direction.value[1]*-1))

	def oppositeDirections(directions: list[Direction]) -> list[Direction]:
		return [Directions.oppositeDirection(direction) for direction in directions]

	def nextPosition(origin: tuple[int, int], direction: Direction) -> tuple[int, int]:
		return (origin[0] + direction.value[0], origin[1] + direction.value[1])

	def nextPositions(origin: tuple[int, int], directions: list[Direction]) -> list[tuple[int, int]]:
		return [(origin[0] + direction.value[0], origin[1] + direction.value[1]) for direction in directions]

class PossiblePlays:
	hasPossiblePlays: bool
	playsList: dict[tuple[int, int]: set[Direction]]

	def __init__(self):
		self.hasPossiblePlays = False
		self.playsList = {}
