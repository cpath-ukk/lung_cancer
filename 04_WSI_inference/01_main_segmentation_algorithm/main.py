'''
Comments to version:
- Uses tissue maps from custom tissue detector (GrandQC tool: paper under review).
Therefore, slides should be processed by tissue detector firstly.
- This is a version with argument parser for slide and output path and slide batch (start:end)
parser.add_argument('--slide_folder', dest='slide_folder', help='path to WSIs', type=str)
parser.add_argument('--output_dir', dest='output_dir', help='path to output folder', type=str)
parser.add_argument('--start', dest='start', help='start num of WSIs', type=int)
parser.add_argument('--end', dest='end', help='end num of WSIs', type=int)
'''

#Define slide batch to process (first 50 slides sorted alphabetically)
#start = 0
#end = 50

###DEVICE
DEVICE = 'cuda'
'''
'cuda:0' - NVIDIA GPU card
'mps'    - APPLE Silicon
'''

###SLIDES
#SLIDE_DIR = '/path/to/slides/'
'''
One folder with single slides.
'''

###OUTPUT (Folder for inference subfolders/files)
#OUTPUT_DIR = '/output/directory/'

###MODEL(S)
#MODEL 1: Tumor detection
MODEL_TUMOR_DIR = './models/lung/'
MODEL_TUMOR_NAME = 'checkpoint.pth'
MPP_MODEL_1 = 1 #define model MPP
M_P_S_MODEL_1 = 512
ENCODER_MODEL_1 = 'timm-efficientnet-b0'
ENCODER_MODEL_1_WEIGHTS = 'imagenet'

# CLASSES
BACK_CLASS = 12 #necessary for generation of final segmentation mask

###MAPS, OVERLAYS: PARAMETERS
OVERLAY_FACTOR = 16 # reduction factor of the overlay compared to dimensions of original WSI

###FLAGS
ONLY_SLIDE_INFO_FLAG = False

###COLORS
from wsi_colors import colors_LUNG as colors

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================
import torch
import argparse
from openslide import open_slide
from PIL import Image
import os
from wsi_slide_info import slide_info
from wsi_process import slide_process_single, slide_process_double
from wsi_maps import make_overlay
import numpy as np
import timeit
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--slide_folder', dest='slide_folder', help='path to WSIs', type=str)
parser.add_argument('--output_dir', dest='output_dir', help='path to output folder', type=str)
parser.add_argument('--start', dest='start', help='start num of WSIs', type=int)
parser.add_argument('--end', dest='end', help='end num of WSIs', type=int)

args = parser.parse_args()

start = args.start
end = args.end
SLIDE_DIR = args.slide_folder
OUTPUT_DIR = args.output_dir

###OUTPUT (TEXT)
case_name = os.path.basename(OUTPUT_DIR)
REPORT_FILE_NAME = f'report_{case_name}_' + str(start) + '_' + str(end)     # File name, ".txt" will be added in the end
REPORT_OUTPUT_DIR = OUTPUT_DIR # where to save the text report

# =============================================================================
# LOAD MODELS
# =============================================================================
if ONLY_SLIDE_INFO_FLAG == False:
    # model_prim = torch.load(MODEL_TUMOR_DIR + MODEL_TUMOR_NAME, map_location=DEVICE)
    model_prim = torch.load(os.path.join(MODEL_TUMOR_DIR, MODEL_TUMOR_NAME), map_location=DEVICE)

# ====================================================================
# PREPARE REPORT FILE, OUTPUT FOLDERS
# =============================================================================

#Prepare report file header
if ONLY_SLIDE_INFO_FLAG == False:
    path_result = os.path.join(REPORT_OUTPUT_DIR, REPORT_FILE_NAME + "_stats_per_slide.txt")
    output_header = "slide_name" + "\t" + "obj_power" + "\t" + "mpp" + "\t"
    output_header = output_header + "patch_n_h_l0" + "\t" + "patch_n_w_l0" + "\t"
    output_header = output_header + "patch_overall" + "\t"
    output_header = output_header + "height" + "\t" + "width" + "\t"
    output_header = output_header + "time"
    output_header = output_header + "\n"
    results = open(path_result, "a+")
    results.write(output_header)
    results.close()

    maps_dir = os.path.join(OUTPUT_DIR, 'maps')
    overlay_dir = os.path.join(OUTPUT_DIR, 'overlays')
    mask_dir = os.path.join(OUTPUT_DIR, 'mask')

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
for slide_name in slide_names[start:end]:
    try:
        #Register start time
        start = timeit.default_timer()

        print("")
        print("Processing:", slide_name)

        # Open slide
        path_slide = os.path.join(SLIDE_DIR, slide_name)
        slide = open_slide(path_slide)

        #GET SLIDE INFO
        p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0, obj_power = slide_info(slide, M_P_S_MODEL_1, MPP_MODEL_1)

        if ONLY_SLIDE_INFO_FLAG:
            continue

        #LOAD TISSUE DETECTION MAP
        tis_det_map = Image.open(os.path.join(OUTPUT_DIR, 'tis_det_mask', slide_name + '_MASK.png'))
        '''
        Tissue detection map is generated on MPP = 10
        This map is used for on-fly control of the necessity of model inference.
        Two variants: reduced version with perfect correlation or full version scaled to working MPP of the tumor detection model
        Classes: 0 - tissue, 1 - background
        '''


        tis_det_map_mpp = np.array(tis_det_map.resize((int(w_l0 * mpp / MPP_MODEL_1), int(h_l0 * mpp / MPP_MODEL_1)), Image.Resampling.LANCZOS))
        '''
        Using original slide dimensions to regenerate tissue detection mask to MPP=2.0 to prevent losses of single 
        pixels and to achieve more concordance to model patch sizes. 
        '''

        map, full_mask = slide_process_single(model_prim, tis_det_map_mpp, slide, patch_n_w_l0, patch_n_h_l0, p_s,
                                                  M_P_S_MODEL_1, colors, ENCODER_MODEL_1,
                                                  ENCODER_MODEL_1_WEIGHTS, DEVICE, BACK_CLASS, MPP_MODEL_1, mpp, w_l0, h_l0)
        stop = timeit.default_timer()
        map_path = os.path.join(maps_dir, slide_name + "_map_TU.png")
        map.save(map_path)

        mask_path = os.path.join(mask_dir, slide_name + "_mask.png")
        cv2.imwrite(mask_path, full_mask)

        del full_mask #to free the memory for parallelization

        overlay = make_overlay(slide, map, p_s, patch_n_w_l0, patch_n_h_l0, OVERLAY_FACTOR)

        del map

        # Save overlaid image
        overlay_im = Image.fromarray(overlay)

        overlay_im_name = os.path.join(overlay_dir, slide_name + "_overlay_TU.jpg")
        overlay_im.save(overlay_im_name)

        del overlay

        # Timer stop
        stop = timeit.default_timer()

        # Write down per slide result
        # Basic data about slide (size, pixel size, objective power, height, width)
        output_temp = slide_name + "\t" + str(obj_power) + "\t" + str(mpp) + "\t"
        output_temp = output_temp + str(patch_n_h_l0) + "\t" + str(patch_n_w_l0) + "\t"
        output_temp = output_temp + str(patch_n_h_l0 * patch_n_w_l0) + "\t"
        output_temp = output_temp + str(patch_n_h_l0 * p_s) + "\t" + str(patch_n_w_l0 * p_s) + "\t"
        output_temp = output_temp + str(round((stop - start) / 60, 1))
        output_temp = output_temp + "\n"
        results = open(path_result, "a+")
        results.write(output_temp)
        results.close()

    except:
        print("There was some problem with the slide.")
