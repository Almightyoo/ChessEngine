import chess
import chess.pgn
import torch
from DataLoader import parse_pgn
from state import ChessState
from train import Net

def evaluate_board(model, board, device):
    state = ChessState(board)
    board_tensor = state.to_tensor().unsqueeze(0).to(device)
    with torch.no_grad():
        evaluation = model(board_tensor)
    return evaluation.item()

def select_best_move(model, board, device):
    best_move = None
    best_evaluation = -float('inf')

    for move in board.legal_moves:
        board.push(move)
        print(move)
        evaluation = evaluate_board(model, board, device)
        print(evaluation)
        board.pop()

        if evaluation > best_evaluation:
            best_evaluation = evaluation
            best_move = move

    return best_move

def play_game_ai_vs_ai(model, device):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    while not board.is_game_over():
        move = select_best_move(model, board, device)
        board.push(move)
        node = node.add_variation(move)

    return game

def play_game_ai_vs_human(model, device):
    board = chess.Board()
    game = chess.pgn.Game()
    node = game

    while not board.is_game_over():
        print(board)
        if board.turn == chess.WHITE:
            move = input("Enter your move (in UCI format, e.g., e2e4): ")
            try:
                board.push_uci(move)
            except ValueError:
                print("Invalid move. Try again.")
                continue
        else:
            move = select_best_move(model, board, device)
            board.push(move)
            print(f"AI plays: {move}")
        
        node = node.add_variation(move)

    return game

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    mode = input("Choose mode (1 for AI vs AI, 2 for AI vs Human): ")

    if mode == '1':
        game = play_game_ai_vs_ai(model, device)
        filename = "self_play_game_1.pgn"
    elif mode == '2':
        game = play_game_ai_vs_human(model, device)
        filename = "human_vs_ai_game.pgn"
    else:
        print("Invalid choice.")
        exit()

    with open(filename, "w") as pgn_file:
        pgn_file.write(str(game))

    print(f"Game finished and saved to {filename}")
