# Dataset paths
CODD_PATH = "/Users/hao/Code/github/inputtest/data/CODD/"
KITTI_PATH = "/data/KITTIodometry/"

INPUT_PATH = '/home/hao/code/github/inputtest/data/inputpoints/'
COORD_PATH = '/home/hao/code/github/inputtest/data/nodecoord/'
OUTPUT_PATH = ('/home/hao/code/github/inputtest/data/Uoutput/')

INPUT_PATH1 = '/home/hao/code/github/inputtest/testdata1/inputpoints/'
COORD_PATH1 = '/home/hao/code/github/inputtest/downsampled/coord/'
OUTPUT_PATH1 = '/home/hao/code/github/inputtest/downsampled/output/'
# Fastreg model parameters
T = 1e-2
VOXEL_SAMPLING_SIZE = 0.3

# Training parameters
lr = 1e-3
batch_size = 6
val_period = 1  # Run validation evaluation every val_period epochs
epochs = 50
