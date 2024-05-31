# Dataset paths
CODD_PATH = "/Users/hao/Code/github/inputtest/data/CODD/"
KITTI_PATH = "/data/KITTIodometry/"

INPUT_PATH = '/home/hao/code/github/inputtest-main/femdata/input/'
COORD_PATH = '/home/hao/code/github/inputtest-main/femdata/'
OUTPUT_PATH = '/home/hao/code/github/inputtest-main/femdata/output/'

INPUT_PATH1 = '/home/hao/code/github/inputtest/testdata/inputpoints/'
COORD_PATH1 = '/home/hao/code/github/inputtest/testdata/output/'
OUTPUT_PATH1 = '/home/hao/code/github/inputtest/testdata/output1/'
# Fastreg model parameters
T = 1e-2
VOXEL_SAMPLING_SIZE = 0.3

# Training parameters
lr = 1e-2
batch_size = 6
val_period = 1  # Run validation evaluation every val_period epochs
epochs = 50
