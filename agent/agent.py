from game.utils import Player, Directions, PossiblePlays
from agent.evaluation import Evaluation
from agent.tree import Knot
import time

class Agent:
	def __init__(self, player: Player, opponent: Player, initialBoard: list[list[Player]], timeLimit: float = 0.95, depthLimit: int = 100, simpleAgent: bool = False):
		self.player = player
		self.opponent = opponent
		self.initialBoard = initialBoard
		self.timeLimit = timeLimit
		self.depthLimit = depthLimit
		self.simpleAgent = simpleAgent

	def choosePlay(self):
		bfBuildTree = time.time()
		root, knotsExpanded, maxDepth = self.buildDecisionTree(self.initialBoard)
		afBuildTree = time.time()
		print(f'Time spent at BuildDecisionTree: {afBuildTree - bfBuildTree}, {knotsExpanded} knots explored, {maxDepth} max depth')

		if (self.simpleAgent):
			bfMinimax = time.time()
			score = self.minimax(root, True)
			afMinimax = time.time()
			print(f'Time spent at Minimax: {afMinimax- bfMinimax}\n')
		else:
			bfAlphaBeta = time.time()
			score = self.alphabeta(root, float("-inf"), float("+inf"), True)
			afAlphaBeta = time.time()
			print(f'Time spent at AlphaBeta: {afAlphaBeta- bfAlphaBeta}\n')

		for child in root.children:
			if child.score == score:
				return child.pos

	def buildDecisionTree(self, board: list[list[Player]]) -> tuple[Knot, int, int]:
		start = time.time()
		knotsExpanded = 0
		maxDepth = 0
		
		root = Knot(board, 0, None, 0)
		queue = [root]
		while len(queue) > 0 and time.time() - start < self.timeLimit: 
			knot = queue.pop(0)
			knotsExpanded += 1
			maxDepth = max(maxDepth, knot.depth)
			isMaximizing = knot.depth % 2 == 0
			player = self.player if isMaximizing else self.opponent
			
			possiblePlays = Agent.possiblePlays(knot.board, player)
			if not possiblePlays.hasPossiblePlays:
				if knot.pos == None:
					continue
				passKnot = Knot(knot.board, self.evaluateBoard(knot.board), None, knot.depth + 1)
				knot.children.append(passKnot)
				if (passKnot.depth < self.depthLimit):
					queue.append(passKnot)
				continue
			
			for pos, directions in possiblePlays.playsList.items():
				
				newBoard = self.applyMove(knot.board, pos, directions, player) 
				newBoardScore = self.evaluateBoard(newBoard)
				
				newKnot = Knot(newBoard, newBoardScore, pos, knot.depth + 1)
				knot.children.append(newKnot)

			knot.children = self.orderMoves(knot.children, isMaximizing)
			if (knot.depth + 1 < self.depthLimit):
				queue.extend(knot.children)

		return root, knotsExpanded, maxDepth
	
	def minimax(self, knot: Knot, isMaximizing: bool):
		if knot.isLeaf():
			return knot.score

		if isMaximizing:
			maxScore = float("-inf")
			for child in knot.children:
				childScore = self.minimax(child, not isMaximizing)
				maxScore = max(maxScore, childScore)

			knot.score = maxScore

		else:
			minScore = float("+inf")
			for child in knot.children:
				childScore = self.minimax(child, not isMaximizing)
				minScore = min(minScore, childScore)

			knot.score = minScore
		
		return knot.score
	
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

	def orderMoves(self, knotChildren: list[Knot], isMaximizing: bool) -> list[Knot]:
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
	
	def evaluateBoard(self, board: list[list[Player]]) -> float:
		totalPieces = 0
		for i in range(8):
			for j in range(8):
				if (board[i][j] != Player.EMPTY): totalPieces += 1
		
		positional = Evaluation.normalize(Evaluation.hPositional(board, self.player, totalPieces), Evaluation.hPositional(board, self.opponent, totalPieces))
		stability = Evaluation.normalize(Evaluation.hStability(board, self.player), Evaluation.hStability(board, self.opponent))
		frontier = Evaluation.normalize(Evaluation.hLoud(board, self.opponent), Evaluation.hLoud(board, self.player))
		corner = Evaluation.normalize(Evaluation.hCorner(board, self.player), Evaluation.hCorner(board, self.opponent))
		pieces = Evaluation.normalize(Evaluation.hPieces(board, self.player), Evaluation.hPieces(board, self.opponent))
		mobility = Evaluation.normalize(len(Agent.possiblePlays(board, self.player).playsList.keys()), len(Agent.possiblePlays(board, self.opponent).playsList.keys()))

		if (self.simpleAgent):
			return pieces
		
		if (totalPieces < 20):
			return (
				(
					7 * positional +
					1 * stability +
					2 * frontier +
					6 * mobility 
				) / 16
			)
		elif (totalPieces < 54):
			return (
				(
					2 * positional +
					3 * stability +
					1 * frontier +
					20 * corner +
					1 * pieces +
					4 * mobility 
				) / 31
			)
		else:
			return (
				(
					4 * stability +
					1 * frontier +
					9 * corner +
					14 * pieces +
					2 * mobility 
				) / 30
			)

	@staticmethod
	def possiblePlays(board: list[list[Player]], player: Player) -> PossiblePlays:
		opponent = Player.BLACK if player == Player.WHITE else Player.WHITE
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