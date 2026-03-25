from game.utils import Player, Direction, Directions, PossiblePlays

class Agent:
	def __init__(self, player: Player, opponent: Player, initialBoard: list[list[Player]]):
		self.player = player
		self.opponent = opponent
		self.initialBoard = initialBoard
	
	def possiblePlays(self, board: list[list[Player]]) -> PossiblePlays:
		plays = PossiblePlays()

		for i in range(len(board)):
			for j in range(len(board[i])):
				if board[i][j] == Player.EMPTY:
					oppDirections = self.searchOpponent((i, j), board)
					if len(oppDirections) > 0:
						for direction in oppDirections:
							if self.foundMyDisc((i, j), direction, board):
								try:
									plays.playsList[(i, j)].add(direction)
								except KeyError:
									plays.playsList[(i, j)] = {direction}

		if len(plays.playsList.keys()) > 0:
			plays.hasPossiblePlays = True
		return plays

	def searchOpponent(self, startPos: tuple[int, int], board: list[list[Player]]) -> list[Directions]:
		foundDirections = []
		for direction in Directions.getAllDirections():
			(i, j) = Directions.nextPosition(startPos, direction)
			if (i >= 0 and j >= 0 and i < 8 and j < 8):
				if (board[i][j] == self.opponent): foundDirections.append(direction)
		return foundDirections
	
	def foundMyDisc(self, startPos: tuple[int, int], direction: Directions, board: list[list[Player]]) -> bool:
		(i, j) = Directions.nextPosition(startPos, direction)
		if (i >= 0 and j >= 0 and i < 8 and j < 8):
			if (board[i][j] == self.opponent): return self.foundMyDisc((i, j), direction, board)
			elif (board[i][j] == self.player): return True
		return False