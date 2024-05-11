import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0 and depth > 0:
        newside, newboard, newflags = makeMove(side, board, moves[0][0], moves[0][1], flags, moves[0][2])
        nextVal, nextMoveList, subMoveTree = minimax(newside, newboard, newflags, depth-1)
        
        value = nextVal
        moveList = [moves[0]] + nextMoveList
        moveTree = {encode(*(moves[0])) : subMoveTree}
        
        for i in range(1,len(moves)):
          newside, newboard, newflags = makeMove(side, board, moves[i][0], moves[i][1], flags, moves[i][2])
          nextVal, nextMoveList, subMoveTree = minimax(newside, newboard, newflags, depth-1)
          if (side and value > nextVal) or ((not side) and value < nextVal):
            value = nextVal
            moveList = [moves[i]] + nextMoveList
          moveTree[encode(*(moves[i]))] = subMoveTree
          
        return (value, moveList, moveTree)
    else:
        return (evaluate(board), [], {})

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) == 0 or depth == 0:
        return (evaluate(board), [], {})
    elif side:
        newside, newboard, newflags = makeMove(side, board, moves[0][0], moves[0][1], flags, moves[0][2])
        nextVal, nextMoveList, subMoveTree = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
        
        value = nextVal
        moveList = [moves[0]] + nextMoveList
        moveTree = {encode(*(moves[0])) : subMoveTree}
        beta = min(beta, value)

        if alpha >= beta:
          return (value, moveList, moveTree)
        
        for i in range(1,len(moves)):
          newside, newboard, newflags = makeMove(side, board, moves[i][0], moves[i][1], flags, moves[i][2])
          nextVal, nextMoveList, subMoveTree = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
          
          if value > nextVal:
            value = nextVal
            moveList = [moves[i]] + nextMoveList
          moveTree[encode(*(moves[i]))] = subMoveTree
          beta = min(beta, value)
          if alpha >= beta:
            return (value, moveList, moveTree)
          
        return (value, moveList, moveTree)
    else:
        newside, newboard, newflags = makeMove(side, board, moves[0][0], moves[0][1], flags, moves[0][2])
        nextVal, nextMoveList, subMoveTree = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
        
        value = nextVal
        moveList = [moves[0]] + nextMoveList
        moveTree = {encode(*(moves[0])) : subMoveTree}
        alpha = max(alpha, value)

        if alpha >= beta:
          return (value, moveList, moveTree)
        
        for i in range(1,len(moves)):
          newside, newboard, newflags = makeMove(side, board, moves[i][0], moves[i][1], flags, moves[i][2])
          nextVal, nextMoveList, subMoveTree = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
          
          if value < nextVal:
            value = nextVal
            moveList = [moves[i]] + nextMoveList
          moveTree[encode(*(moves[i]))] = subMoveTree
          alpha = max(alpha, value)
          if alpha >= beta:
            return (value, moveList, moveTree)
          
        return (value, moveList, moveTree)
    

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    return (evaluate(board), [], {})
