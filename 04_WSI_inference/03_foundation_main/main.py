#For inference only on selected slides from os.sorted list
start = 0
end = 100

###DEVICE
device = 'cuda'
'''
'cuda:0' - NVIDIA GPU card
'mps'    - APPLE Silicon
'''

###SLIDES
SLIDE_DIR = 'path/to/slides'
'''
One folder with single slides.
'''
###OUTPUT (Folder for inference subfolders/files)
OUTPUT_DIR = 'output/dir' #Should contain tissue segmentation masks from main algorithm as only tumor regions are being analyzed
###OUTPUT (TEXT)
REPORT_FILE_NAME = 'report_UNI_DATASET_' +  str(start) + "_" + str(end) # File name, ".txt" will be added in the end.
REPORT_OUTPUT_DIR = OUTPUT_DIR # where to save the text report


###MODEL(S)
#MODEL for subtyping
model = 'uni'
'''
'uni'
'prov-gigapath'
'''
VECTOR_SIZE = 1024
'''
1024 for uni
1536 for prov-gigapath
'''

CLAS_DIR = "models/uni_classifier/"
CLAS_CHECKPOINT = "02_FULL_224_checkpoint_epoch_25.pth"


MPP_MODEL_1 = 1.0
M_P_S_MODEL_1 = 224

#Threshold for classification LUAD vs LUSC
THR = 0.5

###DEFINITION OF OTHER CLASS CODINGS RELEVANT FOR CREATION OF THE FINAL SUBTYPING MASK
BACK_CLASS = 12
TU_CLASS = 1

OVERLAY_FACTOR = 12 # reduction factor of the overlay compared to dimensions of original WSI

###FLAGS
ONLY_SLIDE_INFO_FLAG = False
MAKE_OVERLAY_FLAG = False

###COLORS
from wsi_colors import colors_LUNG_SUB as colors

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import torch
from openslide import open_slide
from PIL import Image
import os
from wsi_slide_info import slide_info
from wsi_process import slide_process_single
from wsi_maps import make_overlay
import numpy as np
import timeit
import cv2
from tqdm import tqdm
import timm
from torchvision import transforms
import torch.nn as nn

# =============================================================================
# LOAD MODELS
# =============================================================================
if model == 'uni':
    model_prim = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    ).to(device)  # Move model to GPU

    model_prim.load_state_dict(torch.load(os.path.join("models/uni", "pytorch_model.bin")), strict=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    model_prim.eval()

if model == 'prov-gigapath':
    model_cfg = {
        "architecture": "vit_giant_patch14_dinov2",
        "num_classes": 0,
        "num_features": 1536,
        "global_pool": "token",
        "model_args": {
            "img_size": 224,
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 1536,
            "depth": 40,
            "num_heads": 24,
            "init_values": 1e-05,
            "mlp_ratio": 5.33334,
            "num_classes": 0
        },
        "pretrained_cfg": {
            "tag": "",
            "custom_load": False,
            "input_size": [
                3,
                224,
                224
            ],
            "fixed_input_size": True,
            "interpolation": "bicubic",
            "crop_pct": 1.0,
            "crop_mode": "center",
            "mean": [
                0.485,
                0.456,
                0.406
            ],
            "std": [
                0.229,
                0.224,
                0.225
            ],
            "num_classes": 0,
            "pool_size": None,
            "first_conv": "patch_embed.proj",
            "classifier": "head",
            "license": "prov-gigapath"
        }
    }

    # Create the model
    model_prim = timm.create_model(
        model_cfg['architecture'],
        num_classes=model_cfg['num_classes'],
        in_chans=model_cfg['model_args']['in_chans'],
        img_size=model_cfg['model_args']['img_size'],
        patch_size=model_cfg['model_args']['patch_size'],
        embed_dim=model_cfg['model_args']['embed_dim'],
        depth=model_cfg['model_args']['depth'],
        num_heads=model_cfg['model_args']['num_heads'],
        init_values=model_cfg['model_args']['init_values'],
        mlp_ratio=model_cfg['model_args']['mlp_ratio']
    )

    weights_path = 'models/prov-gigapath/pytorch_model.bin'
    state_dict = torch.load(weights_path)

    # Load the state dictionary into the model
    model_prim.load_state_dict(state_dict)

    transform = transforms.Compose(
        [
            #transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model_prim.eval()

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_size):
        super(EmbeddingClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model_clas = EmbeddingClassifier(input_size=VECTOR_SIZE).to(device)
# Load the state dictionary from the checkpoint file
checkpoint_path = CLAS_DIR + CLAS_CHECKPOINT
state_dict_clas = torch.load(checkpoint_path, map_location=device)
# Load the state dictionary into the model
model_clas.load_state_dict(state_dict_clas)


# ====================================================================
# PREPARE REPORT FILE, OUTPUT FOLDERS
# =============================================================================

#Prepare report file header
if ONLY_SLIDE_INFO_FLAG == False:
    path_result = REPORT_OUTPUT_DIR + REPORT_FILE_NAME + "_stats_per_slide.txt"
    output_header = "slide_name" + "\t" + "obj_power" + "\t" + "mpp" + "\t"
    output_header = output_header + "patch_n_h_l0" + "\t" + "patch_n_w_l0" + "\t"
    output_header = output_header + "patch_overall" + "\t"
    output_header = output_header + "height" + "\t" + "width" + "\t"
    output_header = output_header + "c_tissue" + "\t"
    output_header = output_header + "c_tumor" + "\t" + "c_mucin" + "\t" + "c_necr" + "\t"
    output_header = output_header + "c_luad" + "\t" + "c_lusc" + "\t"
    output_header = output_header + "perc_luad" + "\t" + "perc_lusc" + "\t"
    output_header = output_header + "subtype" + "\t"
    output_header = output_header + "time"
    output_header = output_header + "\n"
    results = open(path_result, "a+")
    results.write(output_header)
    results.close()

    maps_dir = OUTPUT_DIR + 'maps_pg/'
    overlay_dir = OUTPUT_DIR + 'overlays_pg/'
    mask_dir = OUTPUT_DIR + 'mask_pg/'

    try:
        os.mkdir(maps_dir)
        os.mkdir(overlay_dir)
        os.mkdir(mask_dir)
    except:
        print('The target folders are already there ..')

# ====================================================================
# MAIN SCRIPT
# =============================================================================

#Read in slide names
slide_names = sorted(os.listdir(SLIDE_DIR))
#Start analysis loop
import random
random.seed(42)
random.shuffle(slide_names)

for slide_name in slide_names[start:end]:
    try:
        #Register start time
        if os.path.exists(os.path.join(OUTPUT_DIR, mask_dir, slide_name + '_mask_SUB_F.png')):
             print("mask is already there")
             continue

        start = timeit.default_timer()

        print("")
        print("Processing:", slide_name)

        # Open slide
        path_slide = os.path.join(SLIDE_DIR, slide_name)
        slide = open_slide(path_slide)
        print('loaded slide')
        #GET SLIDE INFO
        p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0, obj_power = slide_info(slide, M_P_S_MODEL_1, MPP_MODEL_1)

        if ONLY_SLIDE_INFO_FLAG:
            continue
        print("Start loading mask")

        #LOAD TISSUE DETECTION MAP
        try:
            tis_det_map = np.array(Image.open(OUTPUT_DIR + "mask/" + slide_name + '_mask.png'))
        except: 
            tis_det_map = np.array(Image.open(OUTPUT_DIR + "mask/" + "C " + slide_name[1:] + '_mask.png'))
        print ("mask loaded")
        buffer_bottom_l = h_l0 - p_s * patch_n_h_l0
        buffer_right_l = w_l0 - p_s * patch_n_w_l0

        # firstly bottom
        #print(end_image.shape)
        buffer_bottom = np.full((int(buffer_bottom_l * mpp / MPP_MODEL_1), tis_det_map.shape[1]), 0)
        tis_det_map = np.concatenate((tis_det_map, buffer_bottom), axis=0)
        # now right side
        #print(end_image.shape)
        temp_image_he, temp_image_wi = tis_det_map.shape  # width and height
        buffer_right = np.full((temp_image_he, int(buffer_right_l * mpp / MPP_MODEL_1)), 0)
        tis_det_map = np.concatenate((tis_det_map, buffer_right), axis=1)

        new_width = int(w_l0 * mpp / MPP_MODEL_1)
        new_height = int(h_l0 * mpp / MPP_MODEL_1)
        print("resizing")
        tis_det_map_mpp = cv2.resize(tis_det_map, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        print("resized")
        map, full_mask = slide_process_single(model_prim, tis_det_map_mpp, slide, patch_n_w_l0, patch_n_h_l0, p_s,
                                              M_P_S_MODEL_1, colors, device, BACK_CLASS, model_clas, THR, transform)

        map_path = maps_dir + slide_name + "_map_SUB_F.png"
        map.save(map_path)

        mask_path = mask_dir + slide_name + "_mask_SUB_F.png"
        cv2.imwrite(mask_path, full_mask)

        count_tis = np.sum (tis_det_map_mpp < 12)
        count_tu = np.sum(tis_det_map_mpp == 1) #tumor
        count_muc = np.sum(tis_det_map_mpp == 4) #mucin
        count_necr = np.sum(tis_det_map_mpp == 3) #necrosis
        count_luad = np.sum(full_mask == 13) #luad
        count_lusc = np.sum(full_mask == 14) #lusc
        count_luad_perc = round(count_luad / (count_lusc + count_luad), 2)
        count_lusc_perc = 1 - count_luad_perc
        if count_luad_perc > 0.5:
            subtype = "luad"
        else:
            subtype = "lusc"

        del full_mask
        del tis_det_map_mpp

        #map.show()
        # =============================================================================
        # 8. MAKE AND SAVE OVERLAY for C8: HEATMAP ON REDUCED AND CROPPED SLIDE CLON
        # =============================================================================

        if MAKE_OVERLAY_FLAG:
            overlay = make_overlay(slide, map, p_s, patch_n_w_l0, patch_n_h_l0, OVERLAY_FACTOR)

            # Save overlaid image
            overlay_im = Image.fromarray(overlay)
            overlay_im_name = overlay_dir + slide_name + "_overlay_SUB_F.jpg"
            overlay_im.save(overlay_im_name)
            del overlay

        del map
        #Calculate square, percent of subtypes
        # Timer stop
        stop = timeit.default_timer()

        # Write down per slide result
        # Basic data about slide (size, pixel size, objective power, height, width)
        output_temp = slide_name + "\t" + str(obj_power) + "\t" + str(mpp) + "\t"
        output_temp = output_temp + str(patch_n_h_l0) + "\t" + str(patch_n_w_l0) + "\t"
        output_temp = output_temp + str(patch_n_h_l0 * patch_n_w_l0) + "\t"
        output_temp = output_temp + str(patch_n_h_l0 * p_s) + "\t" + str(patch_n_w_l0 * p_s) + "\t"
        output_temp  = output_temp  + str(count_tis) + "\t"
        output_temp  = output_temp  + str(count_tu) + "\t" + str(count_muc) + "\t" + str(count_necr) + "\t"
        output_temp  = output_temp  + str(count_luad) + "\t" + str(count_lusc) + "\t"
        output_temp  = output_temp  + str(count_luad_perc) + "\t" + str(count_lusc_perc) + "\t"
        output_temp  = output_temp + subtype + "\t"
        output_temp = output_temp + str(round((stop - start) / 60, 1))

        output_temp = output_temp + "\n"

        results = open(path_result, "a+")
        results.write(output_temp)
        results.close()


    except:
        print('There was a problem with the slide.')
