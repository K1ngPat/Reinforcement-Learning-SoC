import chess
from chess import Move
from node import Node
from edge import Edge
import config
import numpy as np

class MCTS:
    def __init__(self, NN, starting_fen = chess.STARTING_FEN):
        self.trajectory = []
        self.root_node = Node(starting_fen)
        self.NN = NN
        self.trajectory : list[Edge] = []
        self.points = {"a" : 1, "b" : 2, "c" : 3, "d" : 4, "e" : 5, "f" : 6, "g" : 7, "h" : 8}

    def selection(self, curr_node : Node) -> Node:

        while not curr_node.is_leaf_node():
            curr_node.get_child_edges()
            child_edges = curr_node.child_edges
            UCB_vals = [edge.UCB() for edge in child_edges]
            best_edge = child_edges[UCB_vals.index(max(UCB_vals))]
            self.trajectory.append(best_edge)
            curr_node = best_edge.output_node
        return curr_node

    def expansion(self, curr_node : Node) -> Node:
        board = curr_node.board

        possible_actions = list(board.generate_legal_moves())

        if len(possible_actions) == 0:
            outcome = board.outcome(claim_draw=True)
            if outcome is None:
                curr_node.value = 0
            else:
                curr_node.value = 1 if outcome.winner == chess.WHITE else -1
            return curr_node
        
        input_NN = self.neural_input(curr_node.state)
        assert input_NN.shape == (19, 8, 8), f"The shape of output array is {input_NN.shape}"
        p, v = self.NN.predict(input_NN)
        actions = self.get_actions(p, curr_node.state)
        assert1 = 1
        for n_ in p.shape: assert1 *= n_
        assert assert1 == 4672, f"The shape of output array is {p.shape}"
        curr_node.value = v
        curr_node.N += 1

        for action in possible_actions:
            new_state = curr_node.make_move(action)
            curr_node.add_child(Node(new_state), action, actions[action.uci()])
        return curr_node
    
    @staticmethod
    def f(*t) -> np.ndarray:
        output = np.zeros(8,8)
        for pos in t:
            output[pos[0]][pos[1]] = 1
        return output

    def get_piece_positions(board : chess.Board) -> dict[str,list[int]]:
        piece_positions : dict[str,list[int]] = {}

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece = str(piece)
                if piece in piece_positions.keys():
                    piece_positions[piece].append((int(square)//8, int(square)%8))
                else:
                    piece_positions[piece] = [(int(square)//8, int(square)%8)]

        return piece_positions
    
    def neural_input(self, fen : str) -> np.ndarray:
        board = chess.Board(fen)
        piece_positons = self.get_piece_positions(board)
        return np.asarray(np.ones((8,8)) if board.turn == chess.WHITE else np.zeros((8,8)),
                        np.ones((8,8)) if board.has_queenside_castling_rights(chess.WHITE) else np.zeros((8,8)),
                        np.ones((8,8)) if board.has_kingside_castling_rights(chess.WHITE) else np.zeros((8,8)),
                        np.ones((8,8)) if board.has_queenside_castling_rights(chess.BLACK) else np.zeros((8,8)),
                        np.ones((8,8)) if board.has_kingside_castling_rights(chess.BLACK) else np.zeros((8,8)),
                        np.ones((8,8)) if board.can_claim_fifty_moves() else np.zeros((8,8)),
                        self.f(*piece_positons['P']),
                        self.f(*piece_positons['N']),
                        self.f(*piece_positons['R']),
                        self.f(*piece_positons['B']),
                        self.f(*piece_positons['Q']),
                        self.f(*piece_positons['K']),
                        self.f(*piece_positons['p']),
                        self.f(*piece_positons['n']),
                        self.f(*piece_positons['r']),
                        self.f(*piece_positons['b']),
                        self.f(*piece_positons['q']),
                        self.f(*piece_positons['k']),
                        self.f((board.ep_square//8, board.ep_square%8) if board.ep_square is not None else np.zeros((8,8)))
                        )

    def get_actions(self, p : np.ndarray, state) -> dict:
        actions = {}
        board = chess.Board(state)
        p = p.reshape(73, 8, 8)
        board = chess.Board(state)
        valid_moves = list(board.legal_moves)
        
        for moves in valid_moves:
            str_move = moves.uci()
            move = self.map_moves(str_move)
            actions[str_move] = p[move[0]][move[1]][move[2]]
        return actions

    def map_moves(self, move : str) -> tuple:
        # Queen like moves
        move = move.lower()
        check = len(move) == 5
        if check:
            check = check and move[5] != 'q'
            promotion = move[4]
            move = move[:-1]
        move = [self.points[_]-1 if _ in self.points.keys() else int(_) for _ in move]
        x = move[2] - move[0]
        y = move[3] - move[1]
        if check:
            if x==0 and promotion=='r':
                return (64 , move[0], move[1])
            if x==1 and promotion=='r':
                return (65 , move[0], move[1])
            if x==-1 and promotion=='r':
                return (66 , move[0], move[1])
            if x==0 and promotion=='b':
                return (67 , move[0], move[1])
            if x==1 and promotion=='b':
                return (68 , move[0], move[1])
            if x==-1 and promotion=='b':
                return (69 , move[0], move[1])
            if x==0 and promotion=='k':
                return (70 , move[0], move[1])
            if x==1 and promotion=='k':
                return (71 , move[0], move[1])
            if x==-1 and promotion=='k':
                return (72 , move[0], move[1])
        elif abs(x) == abs(y):
            if x>0 and y>0:
                return (x, move[0], move[1])
            elif x>0 and y<0:
                return (7+x, move[0], move[1])
            elif x<0 and y>0:
                return (14-x, move[0], move[1])
            elif x<0 and y<0:
                return (21-x, move[0], move[1])
        elif x==0 and y!=0:
            if y>0:
                return (28+y, move[0], move[1])
            else:
                return (35-y, move[0], move[1])
        elif x!=0 and y==0:
            if x>0:
                return (42+x, move[0], move[1])
            else:
                return (49-x, move[0], move[1])
        elif x==1 and y==2:
            return (56, move[0], move[1])
        elif x==2 and y==1:
            return (57, move[0], move[1])
        elif x==-1 and y==2:
            return (58, move[0], move[1])
        elif x==2 and y==-1:
            return (59, move[0], move[1])
        elif x==-2 and y==1:
            return (60, move[0], move[1])
        elif x==1 and y==-2:
            return (61, move[0], move[1])
        elif x==-1 and y==-2:
            return (62, move[0], move[1])
        elif x==-2 and y==-1:
            return (63, move[0], move[1])

    def backpropogation(self, value : float):
        
        for edge in self.trajectory:
            edge.input_node.N += 1
            edge.N += 1
            edge.value += value

    def train(self):
        for _ in range(config.MCTS_EPOCHS):
            self.trajectory = []

            curr_node = self.selection(self.root_node)
            curr_node.N += 1
            curr_node = self.expansion(curr_node)

            self.backpropogation(curr_node.value)
