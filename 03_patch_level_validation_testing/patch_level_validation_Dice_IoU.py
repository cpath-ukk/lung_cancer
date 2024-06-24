import os
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import segmentation_models_pytorch as smp
from typing import Optional
from tqdm import tqdm

start = 0
end = 100

###PARAMETERS MODEL/NETWORK (necessary to implement specific preprocessing pipeline for network)
ENCODER = 'timm-efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
#DEVICE = 'mps' #APPLE PC
DEVICE = 'cuda' #NVIDIA GPU-based PC: "0" - first (or only) GPU card, "1" - second GPU card.
MODEL_MPP = 1.0 #MPP at which model was trained
M_P_S = 512 #model patch size
###PARAMETERS PATH
MODEL_DIR = '/model/dir/' #multiple models in one folder
BASE_DIR_IMAGE = '/test/dataset/image/' # folder with images
BASE_DIR_GT_MASK = '/test/dataset/mask/' # folder with qupath-style masks
TARGET_DIR = '/output/dir/' # folder to save results.
# Produced is txt file with "model name" "metric counter / class"
TXT_FILE_NAME = 'segmentation_validation_results_' #Type of test (MODE) will be integrated in filename
###CLASSES to calculate Dice/IoU for
CLASSES = [1,2,3,4,5,6,7] #class "0" is ignored (non annotated pixels)
###TEST PARAMETERS
IGNORE_INDEX = 0

#######
#######
###SCRIPT
#######
#######
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def to_tensor_x(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(image,preprocessing_fn):
    image = np.array(image)
    x = preprocessing_fn(image)
    x = to_tensor_x(x)
    return x

def make_class_map (mask, class_colors):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(1, len(class_colors)):
        idx = mask == l
        r[idx] = class_colors [l-1][0]
        g[idx] = class_colors [l-1][1]
        b[idx] = class_colors [l-1][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def get_patch_size (SLIDE_MPP):
    p_s = int(MODEL_MPP / SLIDE_MPP * M_P_S)
    return p_s

def binary_dice_iou_score( #(c) pytorch_toolbelt
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold: Optional[float] = None,
    nan_score_on_empty=False,
    eps: float = 1e-7,
    ignore_index=None,
) -> float:
    """
    Compute IoU score between two image tensors
    :param y_pred: Input image tensor of any shape
    :param y_true: Target image of any shape (must match size of y_pred)
    :param mode: Metric to compute (dice, iou)
    :param threshold: Optional binarization threshold to apply on @y_pred
    :param nan_score_on_empty: If true, return np.nan if target has no positive pixels;
        If false, return 1. if both target and input are empty, and 0 otherwise.
    :param eps: Small value to add to denominator for numerical stability
    :param ignore_index:
    :return: Float scalar
    """
    assert mode in {"dice", "iou"}

    # Make binary predictions
    if threshold is not None:
        y_pred = (y_pred > threshold).to(y_true.dtype)

    if ignore_index is not None:
        mask = (y_true != ignore_index).to(y_true.dtype)
        y_true = y_true * mask
        y_pred = y_pred * mask

    intersection = torch.sum(y_pred * y_true).item()
    cardinality = (torch.sum(y_pred) + torch.sum(y_true)).item()

    if mode == "dice":
        score = (2.0 * intersection) / (cardinality + eps)
    else:
        score = intersection / (cardinality - intersection + eps)

    has_targets = torch.sum(y_true) > 0
    has_predicted = torch.sum(y_pred) > 0

    if not has_targets:
        if nan_score_on_empty:
            score = np.nan
        else:
            score = float(not has_predicted)
    return score

def multiclass_dice_iou_score( #(c) pytorch_toolbelt
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode="dice",
    threshold=None,
    eps=1e-7,
    nan_score_on_empty=False,
    classes_of_interest=None,
    ignore_index=None,
):
    ious = []

    for class_index in classes_of_interest:
        y_pred_i = (y_pred == class_index).float()
        y_true_i = (y_true == class_index).float()
        if ignore_index is not None:
            not_ignore_mask = (y_true != ignore_index).float()
            y_pred_i *= not_ignore_mask
            y_true_i *= not_ignore_mask

        iou = binary_dice_iou_score(
            y_pred=y_pred_i,
            y_true=y_true_i,
            mode=mode,
            nan_score_on_empty=nan_score_on_empty,
            threshold=threshold,
            eps=eps,
        )
        ious.append(iou)

    return ious

####
####
####

MODEL_NAMES = sorted(os.listdir(MODEL_DIR))
image_names = sorted(os.listdir(BASE_DIR_IMAGE))
path_result_iou = TARGET_DIR + TXT_FILE_NAME + '_IoU' + '.txt'
path_result_dice = TARGET_DIR + TXT_FILE_NAME + '_Dice' + '.txt'
output_header = "MODEL" + "\t" + "RESULT" + "\t"
output_header = output_header + "\n"
results = open(path_result_iou, "a+")
results.write(output_header)
results.close()
results = open(path_result_dice, "a+")
results.write(output_header)
results.close()


for MODEL_NAME in MODEL_NAMES [start:end]:
    print('Starting for Model: ', MODEL_NAME)
    model = torch.load(MODEL_DIR + MODEL_NAME, DEVICE)
    global_score_dice = []
    global_score_iou = []
    for image_name in tqdm(image_names, total=len(image_names)):
        image = Image.open (BASE_DIR_IMAGE + image_name)
        mask_gt = Image.open(BASE_DIR_GT_MASK + image_name [:-4] + ".png")
        mask_gt = np.array(mask_gt)

        image_pre = get_preprocessing(image, preprocessing_fn)
        x_tensor = torch.from_numpy(image_pre).to(DEVICE).unsqueeze(0)
        predictions = model.predict(x_tensor)
        predictions = (predictions.squeeze().cpu().numpy())

        #Numpy array [0:512,0:512,1] with pixels = predicted class codes.
        mask_pred = np.argmax(predictions, axis=0).astype('int8')

        y_pred = torch.from_numpy(mask_pred)
        y_true = torch.from_numpy(mask_gt)

        score_iou = multiclass_dice_iou_score(y_pred, y_true, ignore_index=IGNORE_INDEX, mode='iou',
                                              nan_score_on_empty=True, classes_of_interest=CLASSES)
        global_score_iou.append(score_iou)
        score_dice = multiclass_dice_iou_score(y_pred, y_true, ignore_index=IGNORE_INDEX, mode='dice',
                                              nan_score_on_empty=True, classes_of_interest=CLASSES)
        global_score_dice.append(score_dice)

    counter = []
    for y in CLASSES:
        temp = []
        for i in range(len(global_score_iou)):
            if global_score_iou [i] [y-1] > 0:
                temp.append(global_score_iou [i] [y-1])
        counter.append(np.mean(temp))

    output_header = MODEL_NAME + "\t" + str(counter) + "\t"
    output_header = output_header + "\n"
    results = open(path_result_iou, "a+")
    results.write(output_header)
    results.close()

    counter = []
    for y in CLASSES:
        temp = []
        for i in range(len(global_score_dice)):
            if global_score_dice[i][y - 1] > 0:
                temp.append(global_score_dice[i][y - 1])
        counter.append(np.mean(temp))

    output_header = MODEL_NAME + "\t" + str(counter) + "\t"
    output_header = output_header + "\n"
    results = open(path_result_dice, "a+")
    results.write(output_header)
    results.close()
