import chess
from chess import Move
class Node:
    def __init__(self, fen : str) -> None:
        self.state = fen
        self.board = chess.Board(fen)
        self.turn = self.board.turn == 'w'  # True if white's turn
        self.N = 0  # Visit count
        self.value = 0 # Value for this node
        self.check = 0
    
    def get_child_edges(self):
        from edge import Edge
        if not self.check:
            self.check = 1
            legal_moves : list[Move] = list(self.board.legal_moves)    # All the legal moves
            self.child_edges : list["Edge"] = [Edge(self, edge_action) for edge_action in legal_moves]    # List of class "Edge" containing all the edges connected downward in the MCTS tree
            for edge in self.child_edges:
                edge.prob = 1/len(self.child_edges)

    def __eq__(self, other : "Node"):
        return self.state == other.state    

    def make_move(self, action : Move):
        output_board = self.board.copy()
        output_board.push(action)
        output_node = Node(output_board.fen())
        return output_node

    def is_leaf_node(self):
        return self.board.is_game_over(claim_draw=True) or self.N == 0
    
    def add_child(self, action: Move, prob: float):
        from edge import Edge
        edge = Edge(self, action, prob)
        self.child_edges.append(edge)
        return edge