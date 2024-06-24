'''
Comments to version:
- Uses multiclass tissue segmentation maps from main algorithm. Therefore, slides should be processed by tissue detector firstly.
'''

#Analyze only selected files
start = 0
end = 200

###DEVICE
DEVICE = 'cuda'
'''
'cuda:0' - NVIDIA GPU card
'mps'    - APPLE Silicon
'''
###SLIDES
SLIDE_DIR = '/path/to/slides'
'''
One folder with single slides.
'''

###OUTPUT (Folder for inference subfolders/files)
OUTPUT_DIR = '/path/to/output/files/' #Should contain tissue segmentation masks from main algorithm as only tumor regions are being analyzed
###OUTPUT (TEXT)
REPORT_FILE_NAME = 'report_LUNG_SUB_' + str(start) + '_' + str(end)  # File name, ".txt" will be added in the end.
REPORT_OUTPUT_DIR = OUTPUT_DIR # where to save the text report

###MODEL(S)
#MODEL: Subtyping model
MODEL_TUMOR_DIR = './models/lung_sub/'
MODEL_TUMOR_NAME = 'vs04_E28.pth'
MPP_MODEL_1 = 1.0
M_P_S_MODEL_1 = 512
ENCODER_MODEL_1 = 'timm-efficientnet-b0'
ENCODER_MODEL_1_WEIGHTS = 'imagenet'


###OTHER RELEVANT CLASSES FOR PRODUCTION OF THE FINAL SUBTYPING SEGMENTATION MASK
BACK_CLASS = 12
TU_CLASS = 1

OVERLAY_FACTOR = 12 # reduction factor of the overlay compared to dimensions of original WSI

###FLAGS
ONLY_SLIDE_INFO_FLAG = False

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

# =============================================================================
# LOAD MODELS
# =============================================================================
model_prim = torch.load(MODEL_TUMOR_DIR + MODEL_TUMOR_NAME, map_location=DEVICE)

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

    maps_dir = OUTPUT_DIR + 'maps_sub/'
    overlay_dir = OUTPUT_DIR + 'overlays_sub/'
    mask_dir = OUTPUT_DIR + 'mask_sub/'

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
        tis_det_map = np.array(Image.open(OUTPUT_DIR + "mask/" + slide_name + '_mask.png'))
        buffer_bottom_l = h_l0 - p_s * patch_n_h_l0
        buffer_right_l = w_l0 - p_s * patch_n_w_l0

        # firstly bottom
        # print(end_image.shape)
        buffer_bottom = np.full((int(buffer_bottom_l * mpp / MPP_MODEL_1), tis_det_map.shape[1]), 0)
        tis_det_map = np.concatenate((tis_det_map, buffer_bottom), axis=0)
        # now right side
        # print(end_image.shape)
        temp_image_he, temp_image_wi = tis_det_map.shape  # width and height
        buffer_right = np.full((temp_image_he, int(buffer_right_l * mpp / MPP_MODEL_1)), 0)
        tis_det_map = np.concatenate((tis_det_map, buffer_right), axis=1)

        new_width = int(w_l0 * mpp / MPP_MODEL_1)
        new_height = int(h_l0 * mpp / MPP_MODEL_1)

        tis_det_map_mpp = cv2.resize(tis_det_map, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        map, full_mask = slide_process_single(model_prim, tis_det_map_mpp, slide, patch_n_w_l0, patch_n_h_l0, p_s,
                                              M_P_S_MODEL_1, colors, ENCODER_MODEL_1,
                                              ENCODER_MODEL_1_WEIGHTS, DEVICE, BACK_CLASS)

        map_path = maps_dir + slide_name + "_map_TU.png"
        map.save(map_path)

        mask_path = mask_dir + slide_name + "_mask.png"
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
        overlay = make_overlay(slide, map, p_s, patch_n_w_l0, patch_n_h_l0, OVERLAY_FACTOR)

        del map

        # Save overlaid image
        overlay_im = Image.fromarray(overlay)
        overlay_im_name = overlay_dir + slide_name + "_overlay_TU.jpg"
        overlay_im.save(overlay_im_name)

        del overlay

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
