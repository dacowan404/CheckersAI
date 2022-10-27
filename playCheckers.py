from checkers.game import Game
import random
import numpy as np
import tensorflow as tf
import pandas as pd

def convertLocation(position):
    location = [int((position - 1)/4), 2*((position-1)%4)]
    if (location[0] %2==0):
        location[1] += 1
    return location

def convertBoard(game):
    board = np.zeros((8,8))
    for piece in game.board.pieces:
        if piece.position == None:
            continue
        location = [int((piece.position-1)/4), 2*((piece.position-1)%4)]
        if (location[0] %2==0):
            location[1] += 1
        if piece.player == 1 and piece.king:
            board[location[0], location[1]] = -2
        elif piece.player == 1:
            board[location[0], location[1]] = -1
        elif piece.player == 2 and piece.king:
            board[location[0], location[1]] = 2
        else:
            board[location[0], location[1]] = 1
    return board

def printBoard(board):
    length =64
    print(" "*4 + "*"*length)
    print("\t 0\t 1\t 2\t 3\t 4\t 5\t 6\t 7\n")
    print(" " * 4 + "-"*length)
    for i in range(8):
        print(f"{i}||", end="")
        for j in range(8):
            if board[i,j] == 0:
                print("\t ", end="")
            else:
                print(f"\t {board[i,j]}", end="")
        print()
        print("\n" +" "* 4 + "-"*length)
    print(" " * 4 +"-"*length+"\n\n")

def boardMove(board, move):
    nextBoard = np.copy(board)
    startLocation = convertLocation(move[0])
    endLocation = convertLocation(move[1])
    nextBoard[endLocation[0], endLocation[1]] = nextBoard[startLocation[0], startLocation[1]]
    nextBoard[startLocation[0], startLocation[1]] = 0
    return nextBoard
    
def generateAllNextBoards(game, board):
    movesList = game.get_possible_moves()
    nextBoardList = []
    for move in movesList:
        nextBoardList.append(boardMove(board, move))
    return nextBoardList

def switchBoard(board):
    newBoard = np.flip(board, (0,1))
    newBoard = newBoard * -1
    return newBoard
    
def getBestMove(model, game, player=1):
    board = convertBoard(game)
    if player==2:
        board = switchBoard(board)
    nextBoardList = generateAllNextBoards(game, board)
    for count, boardOption in enumerate(nextBoardList):
        boardOption = boardOption.reshape(1,8,8)
        value = model(boardOption).numpy()[0][0]
        if count == 0:
            bestMove = (value, 0)
        elif bestMove[0] < value:
            bestMove = (value, count)
    return bestMove[1]

def generateModel():
    #defines model1 which is 3-layer neural network with sigmoid activation function
    model = tf.keras.Sequential(name="model")
    model.add(tf.keras.Input(shape=(8, 8, 1)))
    model.add(tf.keras.layers.Conv2D(3, 5, activation="sigmoid"))
    model.add(tf.keras.layers.Conv2D(3, 3, activation="sigmoid"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(6, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    
    model.compile()
    #model.summary()
    return model

def updateWeights(model, weights):
    layer0Weights = weights[0:75].reshape((5,5,1,3))
    layer0bias = weights[75:78]
    layer1Weights = weights[78:159].reshape((3,3,3,3))
    layer1bias = weights[159:162]
    layer3Weights = weights[162:234].reshape((12,6))
    layer3bias = weights[234:240]
    layer4Weights = weights[240:246].reshape((6,1))
    layer4bias = weights[246].reshape((1,))
    
    orderedWeights = [[layer0Weights, layer0bias], [layer1Weights, layer1bias], [], [layer3Weights, layer3bias], [layer4Weights, layer4bias]]
    
    for i, layer in enumerate(model.layers):
        if i == 2:
            continue
        layer.set_weights(orderedWeights[i])


def main():
    game = Game()
    weightsDF = pd.read_csv('Gen15WeightsBest.csv', index_col='names')
    model = generateModel()
    bestWeights = weightsDF.loc['Gen15_0111',:].to_numpy()
    updateWeights(model, bestWeights)

    while(not game.is_over()):
    #for i in range(3):
        player = game.whose_turn()
        movesList = game.get_possible_moves()
        if player == 2:
            print("Player 2 Pick a move: ")
            movesList = game.get_possible_moves()
            printBoard(convertBoard(game))
            for number, move in enumerate(movesList):
                start = move[0]
                end = move[1]
                print(f"{number}: {convertLocation(start)}, {convertLocation(end)}")
            print("\nEnter move number:\n ", end="")
            moveInput = input()
        else:
            print("Player 1 moving: ")
            moveInput = getBestMove(model, game)
        moveChoice = movesList[int(moveInput)]
        print(f"Moving pawn at position {moveChoice[0]} to position {moveChoice[1]}")
        game.move(moveChoice)

    print(f"Winner is {game.get_winner()}")

if __name__ == "__main__":
    main()