from game.utils import Player, Directions, PossiblePlays
from tree import Knot
import numpy as np
import time

class Agent:
	def __init__(self, player: Player, opponent: Player, initialBoard: list[list[Player]]):
		self.player = player
		self.opponent = opponent
		self.initialBoard = initialBoard

	def choosePlay(self):
		root = self.buildDecisionTree(self.initialBoard)
		score = self.alphabeta(root, float("-inf"), float("+inf"), True)

		for child in root.children:
			if child.score == score:
				return child.pos

	def buildDecisionTree(self, board: list[list[Player]]) -> Knot:
		start = time.time()
		
		root = Knot(board, 0, None, 0)
		queue = [root]
		while len(queue) > 0 and time.time() - start < 0.5: 
			knot = queue.pop(0)
			isMaximizing = knot.depth % 2 == 0
			
			possiblePlays = self.possiblePlays(board=knot.board)
			if not possiblePlays.hasPossiblePlays:
				continue
			
			for pos, directions in possiblePlays.playsList.items():
				player = self.player if isMaximizing else self.opponent
				
				newBoard = self.applyMove(knot.board, pos, directions, player) 
				newBoardScore = self.evaluateBoard(newBoard)
				
				newKnot = Knot(newBoard, newBoardScore, pos, knot.depth + 1)
				knot.children.append(newKnot)

			knot.children = self.orderMoves(knot.children, isMaximizing)
			queue.extend(knot.children)

		return root
	
	def alphabeta(self, knot: Knot, alpha: float, beta: float, isMaximizing: bool):
		if knot.isLeaf():
			return knot.score

		if isMaximizing:
			maxScore = float("-inf")
			for child in knot.children:
				childScore = self.alphabeta(child, alpha, beta, not isMaximizing)
				maxScore = max(maxScore, childScore)

				alpha = max(alpha, maxScore)

				if alpha >= beta:
					break

			knot.score = maxScore

		else:
			minScore = float("+inf")
			for child in knot.children:
				childScore = self.alphabeta(child, alpha, beta, not isMaximizing)
				minScore = min(minScore, childScore)

				beta = min(beta, minScore)

				if alpha >= beta:
					break

			knot.score = minScore
		
		return knot.score

	def orderMoves(self, knotChildren: list[Knot], isMaximizing: bool) -> list[tuple[int, int]]:
		return sorted(knotChildren, key=lambda knot: knot.score, reverse=isMaximizing)
	
	def applyMove(self, board: list[list[Player]], pos: tuple[int, int], directions: set[Directions], player: Player) -> list[list[Player]]:
		newBoard = [row.copy() for row in board]
		r, c = pos
		newBoard[r][c] = player
		for direction in directions:
			cur = Directions.nextPosition(pos, direction)
			while newBoard[cur[0]][cur[1]] != player:
				newBoard[cur[0]][cur[1]] = player
				cur = Directions.nextPosition(cur, direction)
		return newBoard
	
	def evaluateBoard(self, board: list[list[Player]]) -> int:
		return np.random.randint(0, 100)










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