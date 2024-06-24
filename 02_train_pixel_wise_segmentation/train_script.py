import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import os
from pytorch_toolbelt import losses as L
import numpy as np
import torch
from dataset_v2 import BalancedDataset, CustomBatchDataset, Dataset, get_training_augmentation, get_preprocessing

#PARAMETERS GPU
DEVICE = 'cuda'
TRAIN_PLACE = 'local' #'local' for local gpu station, 'cheops' for training on Cheops supercomputer

#PARAMETERS NETWORK
NET = 'unet++'
'''
'unet' for U-Net, 
'deeplabv3plus' for deeplab V3+
'unet++' for U-Net ++
'''
ENCODER = 'timm-efficientnet-b0'
''' Performance on patch size = 512px and batch size = 8 and U-Net++
'resnet50' 1.7 it/s
'timm-efficientnet-b0' 5.4 it/s
'timm-efficientnet-b1' 4.6 it/s
'timm-efficientnet-b2' 4.4 it/s
'timm-efficientnet-b3' 3.40 it/s
'''


ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
SEED_GLOBAL = 123 #seed for random number generator for batch supply during training (reproducibility)

BATCH_SHUFFLING = True #Shuffle images/batches - normally necessary - otherwise deterministic order of patches
BATCH_SHUFFLING_BEFORE_EACH_EPOCH_FLAG = True #Shuffle batches before each epoch - important for early epoch termination
#when the rarest class is exhausted
EARLY_STOP_EPOCH_RAREST_CLASS_FLAG = True #can not be True if BATCH_SHUFFLING = False
OVERSAMPLE_FLAG = True
CUT_OFF_EMPTY_VALUE = None
TARGET_FILE_NUMBER_OVERSAMPLE = 99999 #define manually, if necessary; normally should be tailored to the most underrepresented classes,
#so that firstly looking at the structure of the dataset using code below will be beneficial

#PARAMETERS DATA
DATA_DIR = '/path/to/dataset'
#Structure:
#train/image train/mask
#val/image val/mask

#PARAMETERS TRAINING
M_P_S = (512,512) #model patch size
N_EPOCHS = 45 #Number epochs
CLASSES = 9999 #number of classes including 0 for non-annotated pixels
B_S_TRAIN = 9999  #Batch size training
B_S_VAL = 1 #Batch size validation
LOSS = 'CE'
''' Available loss function: 
'Lovasz' = Lovasz-softmax, 
'Dice', 
'Focal', 
'CE' = weighted cat cross entropy with global weights (CLASS_WEIGHTS)
'CE_batch_we' = automatic calculation of class weights within single batches (!!! Current implementation for 12 classes)
'Dice_CE_EdgeAware_Loss' = Dice_CE_EdgeAware_Loss
'LovaszSoftmaxLoss' = LovaszSoftmaxLoss
'''
WORKERS_TRAIN = 5  # CPU workers for input pipeline (training) E.g., 4xNumber GPU cards used for training
WORKERS_VAL = 5   # CPU workers for input pipeline (validation)
LR1 = 0.0005 #Initial learning rate
LR_DECOY = [5e-5,5e-6,5e-7,5e-8,5e-5,5e-6,5e-7,5e-5,5e-6,5e-7] #Manual placement of LR decoy
EPOCH_DECOY = [8,12,16,20,24,28,32,35,37,39] #Manual placement of epochs for LR changes
IGNORE_INDEX = 0 #Class code for non-annotated pixels

CLASS_WEIGHTS = [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] #Global class weights. Adjust if necessa

#FRESH MODEL for training on cheops
#NULL_MODEL_NAME = '' #leave unchanged or empty while training on local GPU station

#MODEL NAME TO SAVE CHECKPOINTS
MODEL_SAVE_NAME = '/data/v01/v01_E'

########
######SCRIPT TRAINING
########

x_train_dir = os.path.join(DATA_DIR, 'train/image')
y_train_dir = os.path.join(DATA_DIR, 'train/mask')

x_valid_dir = os.path.join(DATA_DIR, 'val/image')
y_valid_dir = os.path.join(DATA_DIR, 'val/mask')

from torch.utils.data import DataLoader

# create segmentation model with pretrained encoder (Internet to download weights necessary)
if TRAIN_PLACE == 'local':

        if NET == 'unet':
            model = smp.Unet(
                encoder_name=ENCODER,
                encoder_weights=ENCODER_WEIGHTS,
                classes = CLASSES,
                activation=ACTIVATION,
            )
        if NET == 'deeplabv3plus':
            model = smp.DeepLabV3Plus(
                encoder_name=ENCODER,
                encoder_weights=ENCODER_WEIGHTS,
                classes=CLASSES,
                activation=ACTIVATION,
            )

        if NET == 'unet++':
            model = smp.UnetPlusPlus(
                encoder_name=ENCODER,
                encoder_weights=ENCODER_WEIGHTS,
                classes=CLASSES,
                activation=ACTIVATION,
            )

#Save fresh model for cheops on local GPU machine
#torch.save(model, 'Vs_model_unet_resnet50_1024_12cl_fresh.pth')

elif TRAIN_PLACE == 'cheops':
        #Load checkpoint of fresh model if necessary to start training
        model = torch.load(NULL_MODEL_NAME)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

#torch.manual_seed(SEED_GLOBAL)

# Instantiate the balanced dataset
train_balanced_dataset = BalancedDataset(
    x_train_dir,
    y_train_dir,
    model_image_size=M_P_S,
    n_classes=CLASSES,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    method_weights='median',
    cut_off_empty = CUT_OFF_EMPTY_VALUE
    )

train_balanced_dataset.print_info()

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=None,
    preprocessing=get_preprocessing(preprocessing_fn),
    model_image_size=M_P_S
)

# Get the balanced batches
if OVERSAMPLE_FLAG==False:
    batches = train_balanced_dataset.construct_balanced_batches(batch_size=B_S_TRAIN,
                                                               stop_on_rarest=EARLY_STOP_EPOCH_RAREST_CLASS_FLAG,
                                                               shuffle=BATCH_SHUFFLING)
else:
    batches = train_balanced_dataset.get_oversampled_batches(batch_size=B_S_TRAIN,
                                                                stop_on_rarest=EARLY_STOP_EPOCH_RAREST_CLASS_FLAG,
                                                                shuffle=BATCH_SHUFFLING,
                                                                target_file_number_oversample = TARGET_FILE_NUMBER_OVERSAMPLE)


train_balanced_dataset.print_info()

train_balanced_dataset.oversampled_class_weights

# Create a custom sampler based on the balanced batches
train_custom_sampler = CustomBatchDataset(train_balanced_dataset, batches)

# Use the custom sampler with the DataLoader
train_loader = DataLoader(train_custom_sampler, batch_size=None, shuffle=False,
                          num_workers=WORKERS_TRAIN, pin_memory=True)

valid_loader = DataLoader(valid_dataset, batch_size=B_S_VAL, shuffle=False, num_workers=WORKERS_VAL)


if LOSS == 'CE_batch_we':

    from loss_batch_weight import CrossEntropyLoss #Custom batch-weighted version of CrossEntropyLoss
    loss = CrossEntropyLoss(ignore_index=IGNORE_INDEX, weights_on_fly=True, num_classes=CLASSES,
                            device = DEVICE)
    loss.__name__ = 'CrossEntropyLoss'

if LOSS == 'Dice_Focal_Loss':

    from LOSS_edge_aware import Dice_Focal_Loss
    loss = Dice_Focal_Loss(ignore_index=IGNORE_INDEX)
    loss.__name__ = 'DiceFocalLoss'

if LOSS == 'CE':

    from torch import nn
    from segmentation_models_pytorch.utils import base
    class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
        pass
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float)
    loss = CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    #loss.__name__ = 'CrossEntropyLoss'

if LOSS == 'Dice':
    loss = L.DiceLoss(mode='multiclass', ignore_index=IGNORE_INDEX)
    loss.__name__ = 'Dice_loss'

if LOSS == 'Lovasz':
    loss = L.LovaszLoss(ignore=IGNORE_INDEX)
    loss.__name__ = 'Lovasz_loss'

if LOSS == 'Focal':
    loss = L.FocalLoss(ignore_index=IGNORE_INDEX, alpha=0.25,gamma=2) #Hyperparameters from original publication
    loss.__name__ = 'Focal_loss'

if LOSS == 'Dice_CE_EdgeAware_Loss':
    from loss_edge_aware import Dice_CE_EdgeAware_Loss
    loss = Dice_CE_EdgeAware_Loss(ignore_index=IGNORE_INDEX, class_weights=CLASS_WEIGHTS)
    loss.__name__ = 'Dice_CE_EdgeAware_Loss'

if LOSS == 'LovaszSoftmaxLoss_CE':
    from loss_edge_aware import LovaszSoftmaxLoss
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float)
    loss = LovaszSoftmaxLoss_CE(class_weights=class_weights, ignore_index=IGNORE_INDEX)
    loss.__name__ = 'LovaszSoftmaxLoss_CE'

if LOSS == 'BoundaryLoss_CE':
    from loss_edge_aware import LovaszSoftmaxLoss
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float)
    loss = BoundaryLoss_CE(class_weights=class_weights, ignore_index=IGNORE_INDEX)
    loss.__name__ = 'BoundaryLoss_CE'

if LOSS == 'FocalTverskyLoss':
    from loss_edge_aware import FocalTverskyLoss
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float)
    loss = FocalTverskyLoss(class_weights=class_weights, ignore_index=IGNORE_INDEX)
    loss.__name__ = 'FocalTverskyLoss'

if LOSS == 'Lovasz_Hausdorf_CE':
    from loss_edge_aware import Lovasz_Hausdorf_CE
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float)
    loss = Lovasz_Hausdorf_CE(class_weights=class_weights, ignore_index=IGNORE_INDEX)
    loss.__name__ = 'Lovasz_Hausdorf_CE'

if LOSS == 'SobelEdgeLoss_CE':
    from loss_edge_aware import SobelEdgeLoss_CE
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float)
    loss = SobelEdgeLoss_CE(class_weights=class_weights, ignore_index=IGNORE_INDEX)
    loss.__name__ = 'SobelEdgeLoss_CE'

metrics = [
    smp.utils.metrics.IoU(threshold=0.5)
]

metrics_val = [
    utils.metrics.IoU(),
    utils.metrics.Fscore(),
    utils.metrics.Accuracy()
]


optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=LR1),
])

# Create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True
)

valid_epoch = utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics_val,
    device=DEVICE,
    verbose=True
)


torch.backends.cudnn.benchmark = True

for i in range(0, N_EPOCHS):
    print('\nEpoch: {}'.format(i))
    if i > 0:
        if BATCH_SHUFFLING_BEFORE_EACH_EPOCH_FLAG:
            print('Reshuffling the images/batches before starting new epoch..')
            if OVERSAMPLE_FLAG == False:
                batches = train_balanced_dataset.construct_balanced_batches(batch_size=B_S_TRAIN,
                                                                            stop_on_rarest=EARLY_STOP_EPOCH_RAREST_CLASS_FLAG,
                                                                            shuffle=BATCH_SHUFFLING, indices='original')
            else:
                batches = train_balanced_dataset.construct_balanced_batches(batch_size=B_S_TRAIN,
                                                                         stop_on_rarest=EARLY_STOP_EPOCH_RAREST_CLASS_FLAG,
                                                                         shuffle=BATCH_SHUFFLING, indices='oversample')

            # Create a custom sampler based on the balanced batches
            train_custom_sampler = CustomBatchDataset(train_balanced_dataset, batches)

            train_loader = DataLoader(train_custom_sampler, batch_size=None, shuffle=False,
                                      num_workers=WORKERS_TRAIN, pin_memory=True)

    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    if i < 9:
        name = MODEL_SAVE_NAME + str(0) + str(i + 1) + '.pth'
    else:
        name = MODEL_SAVE_NAME + str(i + 1) + '.pth'
    torch.save(model, name)
    print('Model saved!')

    if i == EPOCH_DECOY [0]:
        optimizer.param_groups[0]['lr'] = LR_DECOY [0]
        print('Decrease decoder learning rate to:', LR_DECOY[0])

    if i == EPOCH_DECOY [1]:
        optimizer.param_groups[0]['lr'] = LR_DECOY [1]
        print('Decrease decoder learning rate to:', LR_DECOY[1])

    if i == EPOCH_DECOY [2]:
        optimizer.param_groups[0]['lr'] = LR_DECOY [2]
        print('Decrease decoder learning rate to:', LR_DECOY[2])

    if i == EPOCH_DECOY [3]:
        optimizer.param_groups[0]['lr'] = LR_DECOY [3]
        print('Decrease decoder learning rate to:', LR_DECOY[3])

    if i == EPOCH_DECOY [4]:
        optimizer.param_groups[0]['lr'] = LR_DECOY [4]
        print('Decrease decoder learning rate to:', LR_DECOY[4])

    if i == EPOCH_DECOY [5]:
        optimizer.param_groups[0]['lr'] = LR_DECOY [5]
        print('Decrease decoder learning rate to:', LR_DECOY[5])

    if i == EPOCH_DECOY [6]:
        optimizer.param_groups[0]['lr'] = LR_DECOY [6]
        print('Decrease decoder learning rate to:', LR_DECOY[6])

    if i == EPOCH_DECOY [7]:
        optimizer.param_groups[0]['lr'] = LR_DECOY [7]
        print('Decrease decoder learning rate to:', LR_DECOY[7])

    if i == EPOCH_DECOY [8]:
        optimizer.param_groups[0]['lr'] = LR_DECOY [8]
        print('Decrease decoder learning rate to:', LR_DECOY[8])

    if i == EPOCH_DECOY [9]:
        optimizer.param_groups[0]['lr'] = LR_DECOY [9]
        print('Decrease decoder learning rate to:', LR_DECOY[9])


