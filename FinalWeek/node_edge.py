import chess
from chess import Move
import math

class Node:
    
    def __init__(self, state: str) -> None:
        
        self.state = state
        self.turn = chess.Board(state).turn

        self.edges : list["Edge"] = []

        self.N = 0
        self.value = 0

    def __eq__(self, node: "Node") -> bool:

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
    
class Edge:

    def __init__(self, input_node : Node, output_node : Node, action : Move, prob : float, C_base = 2, C_init = 2):
        
        self.input_node = input_node
        self.output_node = output_node
        self.action = action
        self.prob = prob
        self.C_base = C_base
        self.C_init = C_init

        self.N = 0
        self.value = 0
        self.turn : bool = self.input_node.state.split(" ")[1] == "w"


    def isequal(self, edge : "Edge") -> bool:
        return self.input_node == edge.input_node and self.action == edge.action
    
    
    def UCB(self):
       return (math.log((1 + self.N + self.C_base)/(self.C_base)) + self.C_init) * self.prob * math.sqrt(self.N) / (1 + self.N) + (self.value/(self.N+1) if self.turn else -self.value/(self.N+1))
