import chess.engine
import numpy as np 

def sigmoid(x):
    return 1/(1 + 10**(-x/400))


with chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish") as engine:
    board = chess.Board("8/7p/3P1p2/5B2/1P2Pn2/P5k1/5r2/3R2K1 b - - 14 61")
    info = engine.analyse(board, chess.engine.Limit(depth=12))
    score = info["score"].white()
    mate_score = score.mate()
    centipawn_score = score.score()
    if mate_score:
        print(mate_score)
    else:
        score=2*sigmoid(centipawn_score) - 1
        print(score)
