import chess
from chess import Move
from edge import Edge

class Node:
    
    def __init__(self, state: str) -> None:
        
        self.state = state
        self.turn = chess.Board(state).turn

        self.edges : list["Edge"] = []

        self.N = 0
        self.value = 0

    def isequal(self, node: "Node") -> bool:

        return self.state == node.state

    def move(self, action: Move) -> str:

        board = chess.Board(self.state)
        board.push(action)
        new_state = board.fen()
        del board
        return new_state

    def game_over(self) -> bool:

        board = chess.Board(self.state)
        return board.is_game_over()

    def is_leaf(self) -> bool:
        # Checks if the node is a leaf node
        return self.N == 0

    def add_child(self, child : "Node", action: Move, prob: float) : # -> Edge

        from edge import Edge
        edge = Edge(self, child, action, prob)
        self.edges.append(edge)
        return edge

    def get_all_children(self) -> list["Node"] :

        children = []
        for edge in self.edges:
            children.append(edge.output_node)
            children.extend(edge.output_node.get_all_children())
        return children

    def get_edge_with_action(self, action) : # -> Edge

        for edge in self.edges:
            if edge.action == action:
                return edge
        return None
