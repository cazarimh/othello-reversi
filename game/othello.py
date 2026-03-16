from game.utils import Player, Direction, Directions, PossiblePlays
import numpy as np

class Othello:
	board: list[list[Player]]

	turn: Player
	opponent: Player

	score: dict[str: int]

	winner: Player
	loser: Player
	hasWinner: bool

	directions: list[Direction] = Directions.getAllDirections()

	@staticmethod
	def startGame():
		Othello.setInitialParameters()

		while (not Othello.hasWinner):
			Othello.runRound()
			print(Othello.board)
			Othello.verifyWinner()
			Othello.changeTurn()

		if (Othello.winner != Player.EMPTY):
			print(f'Congratulations, Player {Othello.winner.name}! You won with {Othello.score[Othello.winner.name]}:{Othello.score[Othello.loser.name]} points.')
		else:
			print(f'It\'s a tie!')

		return

	@staticmethod
	def setInitialParameters():
		Othello.board = np.zeros(shape=(8, 8), dtype=np.int8)
		Othello.board[3][4] = Player.BLACK
		Othello.board[4][3] = Player.BLACK
		Othello.board[3][3] = Player.WHITE
		Othello.board[4][4] = Player.WHITE

		Othello.turn = Player.BLACK
		Othello.opponent = Player.WHITE

		Othello.score = {}
		Othello.score['BLACK'] = 2
		Othello.score['WHITE'] = 2

		Othello.winner = Player.EMPTY
		Othello.loser = Player.EMPTY
		Othello.hasWinner = False
		return

	@staticmethod
	def runRound():
		currentPossiblePlays = Othello.possiblePlays()
		if (not currentPossiblePlays.hasPossiblePlays): return

		chosenPos = Othello.displayPossiblePlays(currentPossiblePlays)
		chosenDir = currentPossiblePlays.playsList[chosenPos]
		Othello.propagateChoose(chosenPos, chosenDir)

		return

	@staticmethod
	def possiblePlays() -> PossiblePlays:
		plays = PossiblePlays()

		for i in range(len(Othello.board)):
			for j in range(len(Othello.board[i])):
				if (Othello.board[i][j] == Player.EMPTY):
					oppDirections = Othello.searchOpponent((i, j))
					if (len(oppDirections) > 0):
						for direction in oppDirections:
							if (Othello.foundMyDisc((i, j), direction)):
								try:
									plays.playsList[(i, j)].add(direction)
								except:
									plays.playsList[(i, j)] = {direction}

		if (len(plays.playsList.keys()) > 0): plays.hasPossiblePlays = True
		return plays

	@staticmethod
	def searchOpponent(startPos: tuple[int, int]) -> list[Directions]:
		foundDirections = []
		for direction in Othello.directions:
			(i, j) = Directions.nextPosition(startPos, direction)
			if (i >= 0 and j >= 0 and i < 8 and j < 8):
				if (Othello.board[i][j] == Othello.opponent): foundDirections.append(direction)
		return foundDirections

	@staticmethod
	def foundMyDisc(startPos: tuple[int, int], direction: Directions) -> bool:
		(i, j) = Directions.nextPosition(startPos, direction)
		if (i >= 0 and j >= 0 and i < 8 and j < 8):
			if (Othello.board[i][j] == Othello.opponent): return Othello.foundMyDisc((i, j), direction)
			elif (Othello.board[i][j] == Othello.turn): return True
		return False

	@staticmethod
	def displayPossiblePlays(plays: PossiblePlays) -> tuple[int, int]:
		print(f'\n\n\n\n\n## Player {Othello.turn.name}\'s turn ##\n')
		print(Othello.board)
		playsList = list(plays.playsList.keys())
		for i, (row, col) in enumerate(playsList):
			row += 1
			col = chr(ord('a') + col)
			print(f'{i+1}. {col}{row}')

		while (1):
			try:
				chosen = int(input(f'Choose a play between 1 and {len(playsList)}: '))
				chosen -= 1
			except:
				print('Invalid input. Try again.')
				continue

			print('\n')
			if (chosen >= 0 and chosen < len(playsList)): return playsList[chosen]
		return

	@staticmethod
	def propagateChoose(startPos: tuple[int, int], directions: set[Direction]):
		(i, j) = startPos
		Othello.score[Othello.turn.name] += 1
		Othello.board[i][j] = Othello.turn
		for direction in directions:
			currentPos = Directions.nextPosition(startPos, direction)
			while (Othello.board[currentPos[0]][currentPos[1]] != Othello.turn):
				Othello.board[currentPos[0]][currentPos[1]] = Othello.turn
				Othello.score[Othello.turn.name] += 1
				Othello.score[Othello.opponent.name] -= 1
				currentPos = Directions.nextPosition(currentPos, direction)
		return

	@staticmethod
	def verifyWinner():
		total = Othello.score[Othello.turn.name] + Othello.score[Othello.opponent.name]
		if (total == 64):
			Othello.hasWinner = True
			if (Othello.score[Othello.turn.name] > Othello.score[Othello.opponent.name]):
				Othello.winner = Othello.turn
				Othello.loser = Othello.opponent
			elif (Othello.score[Othello.opponent.name] > Othello.score[Othello.turn.name]):
				Othello.winner = Othello.opponent
				Othello.loser = Othello.turn
			else:
				Othello.winner = Player.EMPTY
				Othello.loser = Player.EMPTY
		return

	@staticmethod
	def changeTurn():
		Othello.turn, Othello.opponent = Othello.opponent, Othello.turn
		return
