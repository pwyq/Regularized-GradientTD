import numpy as np
from RlGlue import BaseEnvironment

# Constants
RIGHT = 0
SKIP = 1

NUM_STATES = 98
NUM_FEATURES = 25
"""
Dyna-Style Planning with Linear Function Approximation and Prioritized Sweeping
https://arxiv.org/pdf/1206.3285.pdf

Our Boyan Chain environment is an extension of that by Boyan (1999, 2002)
from 13 to 98 states, and from 4 to 25 features

Each episode starts at state
N = 98 and terminates in state 0. For all states s > 2,
there is an equal probability of transitioning to states s-1
or s-2 with a reward of -3. From states 2 and 1, there are
deterministic transitions to states 1 and 0 with respective
rewards of -2 and 0. 
"""

class Boyan(BaseEnvironment):
    def __init__(self):
        self.states = NUM_STATES
        self.state = NUM_STATES      # starting state

    def start(self):
        self.state = NUM_STATES
        return self.state

    def step(self, a):
        reward = -3
        terminal = False

        if a == SKIP and self.state > 9:
            print("Double right action is not available in state 10 or state 11... Exiting now.")
            exit()

        if a == RIGHT:
            self.state = self.state + 1
        elif a == SKIP:
            self.state = self.state + 2

        if (self.state == 12):
            terminal = True
            reward = -2

        return (reward, self.state, terminal)

    # def getXPRD(self, target, rep):
    #     N = self.states
    #     # build the state * feature matrix
    #     # add an extra state at the end to encode the "terminal" state
    #     X = np.array([
    #         rep.encode(i) for i in range(N + 1)
    #     ])

    #     # build a transition dynamics matrix
    #     # following policy "target"
    #     P = np.zeros((N + 1, N + 1))
    #     for i in range(11):
    #         P[i, i+1] = .5
    #         P[i, i+2] = .5

    #     P[10, 11] = 1
    #     P[11, 12] = 1

    #     # build the average reward vector
    #     R = np.array([-3] * 10 + [-2, -2, 0])

    #     D = np.diag([0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308])

    #     return X, P, R, D


class BoyanRep:
    def __init__(self):
        # self.map = np.array([
        #     [1,    0,    0,    0   ],
        #     [0.75, 0.25, 0,    0   ],
        #     [0.5,  0.5,  0,    0   ],
        #     [0.25, 0.75, 0,    0   ],
        #     [0,    1,    0,    0   ],
        #     [0,    0.75, 0.25, 0   ],
        #     [0,    0.5,  0.5,  0   ],
        #     [0,    0.25, 0.75, 0   ],
        #     [0,    0,    1,    0   ],
        #     [0,    0,    0.75, 0.25],
        #     [0,    0,    0.5,  0.5 ],
        #     [0,    0,    0.25, 0.75],
        #     [0,    0,    0,    1   ],
        # ])
        self.map = self.generate_map()

    def generate_map(self):
        res = np.zeros(NUM_FEATURES)
        res[0] = 1

        counter = 0
        map = []
        for i in range(NUM_STATES):
            map.append(res.tolist())

            if (res[counter] == 0):
                counter += 1
            if (counter + 1 >= NUM_FEATURES):
                break
            res[counter] -= 0.25
            res[counter+1] += 0.25

        map.append(np.zeros(NUM_FEATURES))  # terminal state - all zeroes vector
        map = np.array(map)                 # convert to np.array
        return map

    def encode(self, s):
        return self.map[s]

    def features(self):
        return NUM_FEATURES
