from dataclasses import dataclass


@dataclass
class CONFIG:
    @dataclass
    class METADATA:
        LOG_PATH = 'work_space'
        SAVE_PATH = 'work_space/save'

    @dataclass
    class DATA:
        ROOT = 'data'
        DATASET = 'vccr' # vccr, ccvid, ccpg
        TRAIN_BATCH = 16
        SAMPLING_STEP = 64
        NUM_WORKERS = 4
        HEIGHT = 256
        WIDTH = 128
        TEST_BATCH = 128
        NUM_INSTANCES = 4

    @dataclass
    class AUG:
        RE_PROB = 0.0
        TEMPORAL_SAMPLING_MODE = 'stride'
        SEQ_LEN = 8
        SAMPLING_STRIDE = 4

    @dataclass
    class MODEL:

        @dataclass
        class AP3D:
            TEMPERATURE = 4
            CONTRACTIVE_ATT = True

        NAME = 'resnet50_attn'
        RES4_STRIDE = 1
        APP_FEATURE_DIM = 2048

    @dataclass
    class LOSS:
        # ID classification loss
        CLA_LOSS = 'crossentropy'
        CLA_LOSS_WEIGHT = 1.
        CLA_S = 16.
        CLA_M = 0.
        # Pairwise loss
        PAIR_LOSS = 'triplet'
        PAIR_LOSS_WEIGHT = 1.
        PAIR_S = 16.
        PAIR_M = 0.3
        # Clothes classification loss
        CLOTHES_CLA_LOSS = 'cosface'
        # Clothes-based adversarial loss
        CAL = 'cal'
        EPSILON = 0.1
        MOMENTUM = 0.

    @dataclass
    class TRAIN:

        @dataclass
        class LR_SCHEDULER:
            STEPSIZE = 20
            DECAY_RATE = 0.1

        @dataclass
        class OPTIMIZER:
            NAME = 'adam'
            LR = 0.0003
            WEIGHT_DECAY = 5e-4

        TYPE = 'pose' # cloth, pose
        TRAIN_MODE = 'standard' # 'one_cloth', 'standard'
        START_EPOCH = 0
        MAX_EPOCH = 60
        RESUME = None # add checkpoint here

    @dataclass
    class TEST:
        TYPE = 'pose' # pose, cloth
        TEST_MODE = 'all' # up, down, front, back, side, front_back, front_side,
        TEST_SET = 'vccr'
