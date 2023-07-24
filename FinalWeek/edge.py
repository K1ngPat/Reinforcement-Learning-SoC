from chess import Move
import math
from node import Node
import config

class Edge:

    def __init__(self, input_node : "Node", action : Move, prob : float = 0):
        
        self.input_node = input_node
        self.action = action
        output_board = self.input_node.board.copy()
        output_board.push(action)
        self.output_node = Node(output_board.fen())
        self.prob = prob

        self.N = 0  # Visit count for this edge
        self.value = 0
        self.turn : bool = self.input_node.state.split(" ")[1] == "w"


    def isequal(self, edge : "Edge") -> bool:
        return self.input_node == edge.input_node and self.action == edge.action
    
    
    def UCB(self):
       return (math.log((1 + self.N + config.C_base)/(config.C_base)) + config.C_init) * self.prob * math.sqrt(self.N) / (1 + self.N) + (self.value/(self.N+1) if self.turn else -self.value/(self.N+1))

    def update_prob(self, new_prob):
        assert new_prob <= 1
        self.prob = new_prob
    
    def update_value(self, new_value):
        self.value = new_value