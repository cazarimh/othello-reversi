from game.utils import Player, Direction, Directions, PossiblePlays
from agent.agent import Agent
import numpy as np
import time


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

    @staticmethod
    def possiblePlays() -> PossiblePlays:
        plays = PossiblePlays()

        for i in range(len(Othello.board)):
            for j in range(len(Othello.board[i])):
                if Othello.board[i][j] == Player.EMPTY:
                    oppDirections = Othello.searchOpponent((i, j))
                    if len(oppDirections) > 0:
                        for direction in oppDirections:
                            if Othello.foundMyDisc((i, j), direction):
                                try:
                                    plays.playsList[(i, j)].add(direction)
                                except:
                                    plays.playsList[(i, j)] = {direction}

        if len(plays.playsList.keys()) > 0:
            plays.hasPossiblePlays = True

        return plays

    @staticmethod
    def searchOpponent(startPos: tuple[int, int]) -> list[Directions]:
        foundDirections = []

        for direction in Othello.directions:
            (i, j) = Directions.nextPosition(startPos, direction)

            if 0 <= i < 8 and 0 <= j < 8:
                if Othello.board[i][j] == Othello.opponent:
                    foundDirections.append(direction)

        return foundDirections

    @staticmethod
    def foundMyDisc(startPos: tuple[int, int], direction: Directions) -> bool:
        (i, j) = Directions.nextPosition(startPos, direction)

        if 0 <= i < 8 and 0 <= j < 8:
            if Othello.board[i][j] == Othello.opponent:
                return Othello.foundMyDisc((i, j), direction)
            elif Othello.board[i][j] == Othello.turn:
                return True

        return False

    @staticmethod
    def propagateChoose(startPos: tuple[int, int], directions: set[Direction]):
        (i, j) = startPos

        Othello.score[Othello.turn.name] += 1
        Othello.board[i][j] = Othello.turn

        for direction in directions:
            currentPos = Directions.nextPosition(startPos, direction)

            while Othello.board[currentPos[0]][currentPos[1]] != Othello.turn:
                Othello.board[currentPos[0]][currentPos[1]] = Othello.turn
                Othello.score[Othello.turn.name] += 1
                Othello.score[Othello.opponent.name] -= 1
                currentPos = Directions.nextPosition(currentPos, direction)

    @staticmethod
    def verifyWinner():
        total = Othello.score[Othello.turn.name] + Othello.score[Othello.opponent.name]

        if total == 64:
            Othello.hasWinner = True

            if Othello.score[Othello.turn.name] > Othello.score[Othello.opponent.name]:
                Othello.winner = Othello.turn
                Othello.loser = Othello.opponent

            elif Othello.score[Othello.opponent.name] > Othello.score[Othello.turn.name]:
                Othello.winner = Othello.opponent
                Othello.loser = Othello.turn

            else:
                Othello.winner = Player.EMPTY
                Othello.loser = Player.EMPTY

    @staticmethod
    def endGameByScore():
        if Othello.score['BLACK'] > Othello.score['WHITE']:
            Othello.winner = Player.BLACK
            Othello.loser = Player.WHITE

        elif Othello.score['WHITE'] > Othello.score['BLACK']:
            Othello.winner = Player.WHITE
            Othello.loser = Player.BLACK

        else:
            Othello.winner = Player.EMPTY
            Othello.loser = Player.EMPTY

        Othello.hasWinner = True

    @staticmethod
    def changeTurn():
        Othello.turn, Othello.opponent = Othello.opponent, Othello.turn