import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)

    # X starts first, so if counts are equal, it's X's turn
    return O if x_count > o_count else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_moves = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_moves.add((i, j))
    return possible_moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    if board[i][j] is not EMPTY:
        raise Exception("Invalid Move: Cell already occupied.")

    # Minimax requires deep copies to explore future states without changing the current board
    new_board = copy.deepcopy(board)
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] is not None:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] is not None:
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] is not None:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] is not None:
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True

    # Check if board is full (Tie)
    for row in board:
        if EMPTY in row:
            return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    current_player = player(board)

    if current_player == X:
        return max_value(board, -math.inf, math.inf)[1]
    else:
        return min_value(board, -math.inf, math.inf)[1]


def max_value(board, alpha, beta):
    if terminal(board):
        return utility(board), None

    v = -math.inf
    best_move = None
    for action in actions(board):
        val, _ = min_value(result(board, action), alpha, beta)
        if val > v:
            v = val
            best_move = action
        alpha = max(alpha, v)
        if beta <= alpha:
            break
    return v, best_move


def min_value(board, alpha, beta):
    if terminal(board):
        return utility(board), None

    v = math.inf
    best_move = None
    for action in actions(board):
        val, _ = max_value(result(board, action), alpha, beta)
        if val < v:
            v = val
            best_move = action
        beta = min(beta, v)
        if beta <= alpha:
            break
    return v, best_move
