C_base = 2000
C_init = 2

MCTS_EPOCHS = 1000

EPOCHS = 100
BATCH_SIZE = 64
N = 8 # board size

num_input_planes =  (2 * 6 + 1) + (1 + 4 + 1)
INPUT_DIM = (BATCH_SIZE, num_input_planes, N, N) # (64, 19, 8, 8)
queen_planes = 56
knight_planes = 8
underpromotion_planes = 9
num_output_planes = queen_planes + knight_planes + underpromotion_planes
OUTPUT_DIM = (BATCH_SIZE, N * N * num_output_planes, 1) # (64, 73 * 8 * 8, 1)

LEARNING_RATE = 0.2
POLICY_WEIGHT = 0.5 # weight of policy loss
VALUE_WEIGHT = 0.5 # weight of value loss
CONVOLUTION_FILTERS = 256
NUM_HIDDEN_LAYERS = 19 # number of residual blocks