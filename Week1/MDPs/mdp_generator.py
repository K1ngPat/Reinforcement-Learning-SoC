import sys
import numpy as np

MAX_NUM_REWARDS = 5
MEAN_REWARD = 0.0
REWARD_STD_DEV = 10.0
PROB_BETA = 2.0

if len(sys.argv) != 3:
    print(f'Usage: {sys.argv[0]} <number of states> <number of actions>')
    exit(-1)

N_s = int(sys.argv[1])
N_a = int(sys.argv[2])
gamma = np.random.random_sample()

with open(f'MDPs/mdp-{N_s}-{N_a}.txt', 'w') as f:
    f.write(f'states {N_s}\n')
    f.write(f'actions {N_a}\n')
    for s in range(N_s):
        for a in range(N_a):
            probs = [{} for _ in range(N_s)]
            probsum = 0.0
            for ss in range(N_s):
                rewards = np.random.randn(np.random.choice(MAX_NUM_REWARDS + 1))
                for r in rewards:
                    probs[ss][r] = np.random.exponential(PROB_BETA)
                    probsum += probs[ss][r]
            if probsum > 0:
                for ss in range(N_s):
                    for r in probs[ss]:
                        f.write(f'tran {s} {a} {ss} {r} {probs[ss][r]/probsum}\n')
    f.write(f'gamma {gamma}')
