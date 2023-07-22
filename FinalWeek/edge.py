from chess import Move
import math
from node import Node

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
