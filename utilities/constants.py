import math

# pyGAME options
WIDTH = 480
HEIGHT = 480
FPS = 120
VISUAL = True
HUMAN_VS_AI = True
NNET_PLAYER = 2

# Neural Network parameters
TRAIN = False
AI_ENGINE = 'hardcode'  # options: random, minimax, hardcode
INPUTS = 27
HIDDEN_LAYERS = [130, 250, 140, 60]
ACTIVATION = 'ReLU'  # options: ReLU, sigmoid, linear
OUTPUTS = 9
OUTPUT_ACTIVATION = 'Linear'  # options: ReLU, sigmoid, linear

# NNet Optimization
NUM_GAMES = 2000
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.0000001
DECAY_RATE = 0.0001
OPTIMIZATION = "ADAM"  # options: vanilla, SGD_momentum, NAG, RMSProp, ADAM
ADAM_BIAS_Correction = True
NAG_COEFF = 0.9
BETA1 = 0.9
BETA2 = 0.999
EPSILON = math.pow(10, -8)

# Cycling Learning Rate
CLR_ON = True
MAX_LR_FACTOR = 10
LR_STEP_SIZE = BATCH_SIZE * 8

# Reinforcement Learning parameters
MEMORY_CAPACITY = 1000000
GAMMA = 0.9
TARGET_UPDATE = 100

# Reward Policy
REWARD_BAD_CHOICE = -15
REWARD_LOST_GAME = -10
REWARD_WON_GAME = 10
REWARD_TIE_GAME = 5
REWARD_NORMALIZATION = False

# epsilon greedy strategy
eSTART = 1
eEND = 0.01
eDECAY = 0.0001

# colors for PyGame
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREY = (185, 185, 185)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)


def one_hot(state):
    one_hot_state = []
    for row in state:
        for cell in row:
            if cell == 0:
                one_hot_state.append(1)
                one_hot_state.append(0)
                one_hot_state.append(0)
            elif cell == 1:
                one_hot_state.append(0)
                one_hot_state.append(1)
                one_hot_state.append(0)
            elif cell == 2:
                one_hot_state.append(0)
                one_hot_state.append(0)
                one_hot_state.append(1)
    return one_hot_state

