from game.utils import Player, Directions, PossiblePlays
from agent.evaluation import Evaluation
from agent.tree import Knot
import time

class Agent:
	def __init__(self, player: Player, opponent: Player, initialBoard: list[list[Player]], timeLimit: float = 0.95, depthLimit: int = 100, simpleAgent: bool = False, baselineAgent: bool = False, minimaxAgent: bool = False):
		self.player = player
		self.opponent = opponent
		self.initialBoard = initialBoard
		self.timeLimit = timeLimit
		self.depthLimit = depthLimit
		self.simpleAgent = simpleAgent
		self.baselineAgent = baselineAgent
		self.minimaxAgent = minimaxAgent

	def choosePlay(self):
		print(f'{self.player.name} playing')

		bfItDeep = time.time()
		move = self.iterativeDeepening(self.initialBoard)
		afItDeep = time.time()
		print(f'Time spent at Iterative Deepening: {afItDeep - bfItDeep}, {move} chosen\n')

		return move

	def iterativeDeepening(self, board: list[list[Player]]) -> tuple[int, int]:
		start = time.time()
		self.maxDepth = 0
		self.knotsExpanded = 0

		bestMove = None

		for limit in range(1, self.depthLimit + 1):
			if (time.time() - start > self.timeLimit):
				return bestMove

			root = Knot(board, self.evaluateBoard(board), None, 0)

			if not self.minimaxAgent:
				bfAlphabeta = time.time()
				score, move, expanded, pruned, timedOut = self.alphabeta(root, float("-inf"), float("+inf"), start, limit, True)
				afAlphabeta = time.time()
				print(f'AlphaBeta:')
				print(f'Depth {limit} completed in {afAlphabeta - bfAlphabeta}s')
				print(f'{expanded} nodes expanded and {pruned} nodes pruned')
				print(f'Found {score} score for {move} move')

			else:
				bfMinimax = time.time()
				score, move, expanded, timedOut = self.minimax(root, start, limit, True)
				afMinimax = time.time()
				print(f'Minimax:')
				print(f'Depth {limit} completed in {afMinimax - bfMinimax}s')
				print(f'{expanded} nodes expanded')
				print(f'Found {score} score for {move} move')

			if timedOut:
				break

			bestMove = move
			self.maxDepth = limit
			self.knotsExpanded += expanded

		if (bestMove == None):
			return (list(Agent.possiblePlays(board, self.player).playsList.keys()))[-1]

		return bestMove

	def generateChildren(self, father: Knot, player: Player) -> list[Knot]:
		isMaximizing = True if player == self.player else False

		plays = Agent.possiblePlays(father.board, player)

		children = []

		if not plays.hasPossiblePlays:
			return children

		for move, directions in plays.playsList.items():
			newBoard = self.applyMove(father.board, move, directions, player)
			newScore = self.evaluateBoard(newBoard)

			rootMove = father.pos if father.pos is not None else move
			newKnot = Knot(newBoard, newScore, rootMove, father.depth + 1)
			children.append(newKnot)

		return self.orderMoves(children, isMaximizing)

	def minimax(self, knot: Knot, startTime: float, depthLimit: int, isMaximizing: bool, lastPassed: bool = False):
		if time.time() - startTime >= self.timeLimit:
			return (self.evaluateBoard(knot.board), knot.pos, 1, True)

		if depthLimit == 0:
			return (self.evaluateBoard(knot.board), knot.pos, 1, False)

		player = self.player if isMaximizing else self.opponent
		children = self.generateChildren(knot, player)
		passTurn = len(children) == 0
		knot.children = children

		if passTurn:
			if lastPassed:
				return (self.evaluateBoard(knot.board), knot.pos, 1, False)
			return self.minimax(knot, startTime, depthLimit-1, not isMaximizing, passTurn)

		totalExp = 0
		bestMove = None

		if isMaximizing:
			maxScore = float("-inf")
			for child in knot.children:
				score, move, exp, timedOut = self.minimax(child, startTime, depthLimit-1, not isMaximizing, passTurn)
				if score > maxScore:
					maxScore = score
					bestMove = move

				if timedOut:
					return (maxScore, bestMove, totalExp + exp, True)

				totalExp += exp

			knot.score = maxScore

		else:
			minScore = float("+inf")
			for child in knot.children:
				score, move, exp, timedOut = self.minimax(child, startTime, depthLimit-1, not isMaximizing, passTurn)
				if score < minScore:
					minScore = score
					bestMove = move

				if timedOut:
					return (minScore, bestMove, totalExp + exp, True)

				totalExp += exp

			knot.score = minScore

		return (knot.score, bestMove, totalExp + 1, False)

	def alphabeta(self, knot: Knot, alpha: float, beta: float, startTime: float, depthLimit: int, isMaximizing: bool, lastPassed: bool = False):
		if time.time() - startTime >= self.timeLimit:
			return (self.evaluateBoard(knot.board), knot.pos, 1, 0, True)

		if depthLimit == 0:
			return (knot.score, knot.pos, 1, 0, False)

		player = self.player if isMaximizing else self.opponent
		children = self.generateChildren(knot, player)
		passTurn = len(children) == 0
		knot.children = children

		if passTurn:
			if lastPassed:
				return (knot.score, knot.pos, 1, 0, False)
			return self.alphabeta(knot, alpha, beta, startTime, depthLimit-1, not isMaximizing, passTurn)

		totalExp = 0
		totalPrun = 0
		bestMove = None

		if isMaximizing:
			maxScore = float("-inf")
			for i, child in enumerate(knot.children):
				score, move, exp, prun, timedOut = self.alphabeta(child, alpha, beta, startTime, depthLimit-1, not isMaximizing, passTurn)
				if timedOut:
					return (score, move, totalExp + exp, totalPrun + prun, True)

				if score > maxScore:
					maxScore = score
					bestMove = move

				totalExp += exp
				totalPrun += prun

				alpha = max(alpha, maxScore)

				if alpha >= beta:
					totalPrun += len(knot.children) - i - 1
					break

			knot.score = maxScore

		else:
			minScore = float("+inf")
			for i, child in enumerate(knot.children):
				score, move, exp, prun, timedOut = self.alphabeta(child, alpha, beta, startTime, depthLimit-1, not isMaximizing, passTurn)
				if timedOut:
					return (score, move, totalExp + exp, totalPrun + prun, True)

				if score < minScore:
					minScore = score
					bestMove = move

				totalExp += exp
				totalPrun += prun

				beta = min(beta, minScore)

				if alpha >= beta:
					totalPrun += len(knot.children) - i - 1
					break

			knot.score = minScore

		return (knot.score, bestMove, totalExp + 1, totalPrun, False)

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

		if (self.baselineAgent):
			return pieces

		elif (self.simpleAgent):
			return (
				3 * positional +
				2 * stability +
				1 * frontier +
				5 * corner +
				3 * pieces +
				5 * mobility
			)

		else:
			if (totalPieces < 20):
				return (
						8 * mobility +
						6 * positional +
						3 * frontier
				)
			elif (totalPieces < 54):
				return (
						10 * corner +
						6 * mobility +
						4 * stability +
						2 * positional +
						1 * frontier
				)
			else:
				return (
						10 * pieces +
						6 * corner +
						3 * stability +
						1 * mobility
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