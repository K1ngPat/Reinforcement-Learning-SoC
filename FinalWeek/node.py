import chess
from chess import Move
class Node:
    def __init__(self, fen : str) -> None:
        self.state = fen
        self.board = chess.Board(fen)
        self.turn = self.board.turn == 'w'  # True if white's turn
        self.N = 0  # Visit count
        self.value = 0 # Value for this node
    
    def get_child_edges(self):
        from edge import Edge
        self.legal_moves : list[Move] = self.board.legal_moves    # All the legal moves
        self.child_edges : list["Edge"] = [Edge(self, edge_action) for edge_action in self.legal_moves]    # List of class "Edge" containing all the edges connected downward in the MCTS tree

    def update_value(self, new_value):
        self.value = new_value