from game.utils import Player, Directions, Direction as d, BoardHouses as h

class Evaluation():
  scoreBoard = [[h.CORNER, h.C, h.A, h.B, h.B, h.A, h.C, h.CORNER],
                [h.C, h.X, h.SIMPLE, h.SIMPLE, h.SIMPLE, h.SIMPLE, h.X, h.C],
                [h.A, h.SIMPLE, h.DOUBLE, h.DOUBLE, h.DOUBLE, h.DOUBLE, h.SIMPLE, h.A],
                [h.B, h.SIMPLE, h.DOUBLE, h.DOUBLE, h.DOUBLE, h.DOUBLE, h.SIMPLE, h.B],
                [h.B, h.SIMPLE, h.DOUBLE, h.DOUBLE, h.DOUBLE, h.DOUBLE, h.SIMPLE, h.B],
                [h.A, h.SIMPLE, h.DOUBLE, h.DOUBLE, h.DOUBLE, h.DOUBLE, h.SIMPLE, h.A],
                [h.C, h.X, h.SIMPLE, h.SIMPLE, h.SIMPLE, h.SIMPLE, h.X, h.C],
                [h.CORNER, h.C, h.A, h.B, h.B, h.A, h.C, h.CORNER]]

  @staticmethod
  def hPositional(board: list[list[Player]], player: Player) -> float:
    positionalScore = 0
    for i in range(8):
      for j in range(8):
        if (board[i][j] == player):
          positionalScore += Evaluation.scoreBoard[i][j]

    if (board[0][0] == player):
      if (board[1][1] == player): positionalScore += -h.X
      if (board[0][1] == player): positionalScore += -h.C
      if (board[0][2] == player): positionalScore += -h.A
      if (board[0][3] == player): positionalScore += -h.B
      if (board[1][0] == player): positionalScore += -h.C
      if (board[2][0] == player): positionalScore += -h.A
      if (board[3][0] == player): positionalScore += -h.B

    if (board[0][7] == player):
      if (board[1][6] == player): positionalScore += -h.X
      if (board[0][6] == player): positionalScore += -h.C
      if (board[0][5] == player): positionalScore += -h.A
      if (board[0][4] == player): positionalScore += -h.B
      if (board[1][7] == player): positionalScore += -h.C
      if (board[2][7] == player): positionalScore += -h.A
      if (board[3][7] == player): positionalScore += -h.B

    if (board[7][0] == player):
      if (board[6][1] == player): positionalScore += -h.X
      if (board[6][0] == player): positionalScore += -h.C
      if (board[5][0] == player): positionalScore += -h.A
      if (board[4][0] == player): positionalScore += -h.B
      if (board[7][1] == player): positionalScore += -h.C
      if (board[7][2] == player): positionalScore += -h.A
      if (board[7][3] == player): positionalScore += -h.B

    if (board[7][7] == player):
      if (board[6][6] == player): positionalScore += -h.X
      if (board[7][6] == player): positionalScore += -h.C
      if (board[7][5] == player): positionalScore += -h.A
      if (board[7][4] == player): positionalScore += -h.B
      if (board[6][7] == player): positionalScore += -h.C
      if (board[5][7] == player): positionalScore += -h.A
      if (board[4][7] == player): positionalScore += -h.B

    return positionalScore / 0.88

  @staticmethod
  def hLoud(board: list[list[Player]], player: Player) -> float:
    frontierPieces = 0
    for i in range(8):
      for j in range(8):
        if (board[i][j] == player):
          originalDir = [d.N, d.E, d.S, d.W]
          dir1 = Directions.nextPositions((i, j), originalDir)
          dir2 = Directions.nextPositions((i, j), Directions.oppositeDirections(originalDir))
          for k in range(4):
            (n, m) = dir1[k]
            (p, q) = dir2[k]
            if (n >= 0 and m >= 0 and n < 8 and m < 8 and
                p >= 0 and q >= 0 and p < 8 and q < 8):
              if ((board[n][m] == Player.EMPTY and board[p][q] != Player.EMPTY) or
                  (board[n][m] != Player.EMPTY and board[p][q] == Player.EMPTY)):
                frontierPieces += 1

    return frontierPieces / 0.64

  @staticmethod
  def hPieces(board: list[list[Player]], player: Player) -> float:
    playerPieces = 0
    for i in range(8):
      for j in range(8):
        if (board[i][j] == player):
          playerPieces += 1

    return playerPieces
