import os
import chess.pgn
import chess.engine
import torch
import numpy as np
from state import ChessState

path = "KingBase2019-pgn"

def sigmoid(x):
    return 1 / (1 + 10 ** (-x / 400))

def parse_pgn(path, max_games):
    X = []
    Y = []
    game_count = 0
    
    with chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish") as engine:
        for fn in os.listdir(path):
            if fn.endswith(".pgn"):
                fp = os.path.join(path, fn)
                with open(fp) as pgn:
                    while game_count < max_games:
                        game = chess.pgn.read_game(pgn)
                        if not game:
                            break

                        board = game.board()
                        for move in game.mainline_moves():
                            board.push(move)
                            # print(board)
                            state = ChessState(board)
                            board_tensor = state.to_tensor()
                            X.append(board_tensor)

                            info = engine.analyse(board, chess.engine.Limit(depth=5))
                            score = info["score"].white()
                            mate_score = score.mate()
                            if mate_score is not None:
                                if mate_score<0:
                                    eval=-1
                                else:
                                    eval=1
                            else:
                                centipawn_score = score.score()
                                eval = 2 * sigmoid(centipawn_score) - 1
                            # print(eval)
                            Y.append(eval)
                        
                        game_count += 1
                        print(game_count)
                        if game_count >= max_games:
                            break 
                            X = np.array(X)
    X= np.array(X)
    Y= np.array(Y)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    return X, Y

# X, Y = parse_pgn(path, 15)
# print(X.shape,Y.shape)
