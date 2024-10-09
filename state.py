import chess
import torch

class ChessState:
    def __init__(self, board):
        self.board = board
        self.piece_map = self.board.piece_map()

    def to_tensor(self):
        board_tensor = torch.zeros(14, 8, 8)  # Updated to 14 channels
        piece_index = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }

        # Populate the first 12 channels with the piece positions
        for square, piece in self.piece_map.items():
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            piece_type = piece_index[piece.piece_type]
            channel = piece_type if piece.color == chess.WHITE else piece_type + 6
            board_tensor[channel, 7 - rank, file] = 1 if piece.color == chess.WHITE else -1

        # Populate the 13th channel with squares attacked by white pieces
        for square in chess.SQUARES:
            if self.board.is_attacked_by(chess.WHITE, square):
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                board_tensor[12, 7 - rank, file] = -1  # White attacks

        # Populate the 14th channel with squares attacked by black pieces
        for square in chess.SQUARES:
            if self.board.is_attacked_by(chess.BLACK, square):
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                board_tensor[13, 7 - rank, file] = 1  # Black attacks

        return board_tensor

# Example usage
board = chess.Board("r2q1rk1/ppp2ppp/2nb1n2/1B3b2/8/2N1PN2/PPP2PPP/R1BQ1RK1 b - - 4 8")
state = ChessState(board)
board_tensor = state.to_tensor()
print(board_tensor)

