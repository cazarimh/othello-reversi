from game.utils import Player, Directions, PossiblePlays
from agent.evaluation import Evaluation
from agent.tree import Knot
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
		while len(queue) > 0 and time.time() - start < 0.95: 
			knot = queue.pop(0)
			isMaximizing = knot.depth % 2 == 0
			
			possiblePlays = Agent.possiblePlays(knot.board, self.player,  self.opponent)
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
		totalPieces = 0
		for i in range(8):
			for j in range(8):
				if (board[i][j] != Player.EMPTY): totalPieces += 1
		
		positional = Evaluation.hPositional(board, self.player) - Evaluation.hPositional(board, self.opponent)
		stability = Evaluation.hStability(board, self.player) - Evaluation.hStability(board, self.opponent)
		frontier = Evaluation.hLoud(board, self.opponent) - Evaluation.hLoud(board, self.player)
		corner = Evaluation.hCorner(board, self.player) - Evaluation.hCorner(board, self.opponent)
		pieces = Evaluation.hPieces(board, self.player) - Evaluation.hPieces(board, self.opponent)
		mobility = len(Agent.possiblePlays(board, self.player, self.opponent).playsList.keys()) - len(Agent.possiblePlays(board, self.player, self.opponent).playsList.keys())
		
		if (totalPieces < 20):
			return (
				(
					4 * positional +
					1 * stability +
					2 * frontier +
					3 * corner +
					4 * mobility
				) / 14
			)
		elif (totalPieces < 52):
			return (
				(
					2 * positional +
					5 * stability +
					2 * frontier +
					6 * corner +
					2 * pieces +
					4 * mobility
				) / 21
			)
		else:
			return (
				(
					2 * stability +
					1 * frontier +
					8 * pieces +
					2 * mobility
				) / 13
			)

	@staticmethod
	def possiblePlays(board: list[list[Player]], player: Player, opponent: Player) -> PossiblePlays:
		plays = PossiblePlays()

		for i in range(len(board)):
			for j in range(len(board[i])):
				if board[i][j] == Player.EMPTY:
					oppDirections = Agent.searchOpponent((i, j), board, opponent)
					if len(oppDirections) > 0:
						for direction in oppDirections:
							if Agent.foundMyDisc((i, j), direction, board, player, opponent):
								try:
									plays.playsList[(i, j)].add(direction)
								except KeyError:
									plays.playsList[(i, j)] = {direction}

		if len(plays.playsList.keys()) > 0:
			plays.hasPossiblePlays = True
		return plays

	@staticmethod
	def searchOpponent(startPos: tuple[int, int], board: list[list[Player]], opponent: Player) -> list[Directions]:
		foundDirections = []
		for direction in Directions.getAllDirections():
			(i, j) = Directions.nextPosition(startPos, direction)
			if (i >= 0 and j >= 0 and i < 8 and j < 8):
				if (board[i][j] == opponent): foundDirections.append(direction)
		return foundDirections
	
	@staticmethod
	def foundMyDisc(startPos: tuple[int, int], direction: Directions, board: list[list[Player]], player: Player, opponent: Player) -> bool:
		(i, j) = Directions.nextPosition(startPos, direction)
		if (i >= 0 and j >= 0 and i < 8 and j < 8):
			if (board[i][j] == opponent): return Agent.foundMyDisc((i, j), direction, board, player, opponent)
			elif (board[i][j] == player): return True
		return False