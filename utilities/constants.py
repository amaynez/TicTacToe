import math

# pyGAME options
WIDTH = 480
HEIGHT = 480
FPS = 120
VISUAL = False
PLAY_VS_AI = False
NNET_PLAYER = 2

# Neural Network parameters
TRAIN = True
RANDOM_AI = True
INPUTS = 9
HIDDEN_LAYERS = [200, 200]
OUTPUTS = 9
ACTIVATION = 'ReLU'  # options: ReLU, sigmoid, linear
OUTPUT_ACTIVATION = 'Linear'  # options: ReLU, sigmoid, linear

# NNet Optimization
NUM_EPISODES = 10000
BATCH_SIZE = 1024
LEARNING_RATE = 0.003
OPTIMIZATION = "ADAM"  # options: vanilla, SGD_momentum, NAG, RMSProp, ADAM
ADAM_BIAS_Correction = False
NAG_COEFF = 0.9
DECAY_RATE = 0.01
GAMMA_OPT = 0.9
BETA = 0.999
EPSILON = math.pow(10, -8)

# Cycling Learning Rate
CLR_ON = True
MAX_LR_FACTOR = 6
LR_STEP_SIZE = BATCH_SIZE * 8

# Reinforcement Learning parameters
MEMORY_CAPACITY = 100000
GAMMA = 0.99
TARGET_UPDATE = 8

# Reward Policy
REWARD_BAD_CHOICE = -5
REWARD_LOST_GAME = -5
REWARD_WON_GAME = 5
REWARD_TIE_GAME = 5

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
