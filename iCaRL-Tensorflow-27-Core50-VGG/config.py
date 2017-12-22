'''
This module reads the configuration from command line and makes it available my global variables.
'''

import argparse
import sys
# Default Settings
nb_proto = 50
num_classes = 50                # Total number of classes
num_classes_itera = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Total number of classes for each iteration
batch_size = 256                # Batch size
nb_batches = 9                  # Number of groups
epochs_first_batch = 5          # Training epochs in first batch
epochs_other_batches = 5        # Training epochs in other batches
initial_lr_first_batch = 0.04   # Initial learning rate for first batch
lr_strat_first_batch = []       # Epochs where learning rate gets decreased for first batch
initial_lr_other_batches = 0.08 # Initial learning rate in batches other than the first
lr_strat_other_batches = []     # Epochs where learning rate gets decreased in batches other than the first
lr_factor = 5.                  # Learning rate decrease factor
gpu = '0'                       # Used GPU
wght_decay = 0.0005             # Weight Decay
momentum = 0.9                  # Momentum for SGD
image_size = 128
network = 'mvggnet'

# Parse command line arguments
parser = argparse.ArgumentParser(description='iCaRL running on Core50 dataset')
parser.add_argument('--run', required=True, help='The run to execute')
parser.add_argument('--lr_1', type=float, help='Learning rate for the first batch')
parser.add_argument('--ep_1', type=int, help='Learning epochs for the first batch')
parser.add_argument('--lr_o', type=float, help='Learning rate for the other batches')
parser.add_argument('--ep_o', type=int, help='Learning epochs for the other batches')
parser.add_argument('--stored_images', type=int, help='Stored images per class')
parser.add_argument('--weight_decay', type=float, help='Weight decay')
parser.add_argument('--image_size', type=int, help='Images size as network input')
parser.add_argument('--network', help='Network: mvggnet or mcaffenet')
args = parser.parse_args()

# Override some settings
execution = args.run
epochs_first_batch = args.ep_1 if args.ep_1 is not None else epochs_first_batch
initial_lr_first_batch = args.lr_1 if args.lr_1 is not None else initial_lr_first_batch
epochs_other_batches = args.ep_o if args.ep_o is not None else epochs_other_batches
initial_lr_other_batches = args.lr_o if args.lr_o is not None else initial_lr_other_batches
nb_proto = args.stored_images if args.stored_images is not None else nb_proto
wght_decay = args.weight_decay if args.weight_decay is not None else wght_decay
image_size = args.image_size if args.image_size is not None else image_size
network = args.network if args.network is not None else network

if network != 'mvggnet' and network != 'mcaffenet':
    sys.exit(1)