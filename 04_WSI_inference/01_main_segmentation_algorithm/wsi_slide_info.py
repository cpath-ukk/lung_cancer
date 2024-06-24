# EXTRACTION OF META-DATA FROM SLIDE
from PIL import Image
import numpy as np


def slide_info(slide, m_p_s, mpp_model):
    # Objective power
    try:
        obj_power = slide.properties["openslide.objective-power"]
    except:
        obj_power = 99

    # Microne per pixel
    mpp = float(slide.properties["openslide.mpp-x"])
    p_s = int(mpp_model / mpp * m_p_s)

    # Vendor
    vendor = slide.properties["openslide.vendor"]

    # Extract and save dimensions of level [0]
    dim_l0 = slide.level_dimensions[0]
    w_l0 = dim_l0[0]
    h_l0 = dim_l0[1]

    # Calculate number of patches to process
    patch_n_w_l0 = int(w_l0 / p_s)
    patch_n_h_l0 = int(h_l0 / p_s)

    # Number of levels
    num_level = slide.level_count

    # Level downsamples
    down_levels = slide.level_downsamples

    # Output BASIC DATA
    print("")
    print("Basic data about processed whole-slide image")
    print("")
    print("Vendor: ", vendor)
    print("Scan magnification: ", obj_power)
    print("Number of levels: ", num_level)
    print("Level downsamples: ", down_levels)
    print("Microns per pixel (slide):", mpp)
    print("Height: ", h_l0)
    print("Width: ", w_l0)
    print("Model patch size at slide MPP: ", p_s, "x", p_s)
    print("Width - number of patches: ", patch_n_w_l0)
    print("Height - number of patches: ", patch_n_h_l0)
    print("Overall number of patches / slide (without tissue detection): ", patch_n_w_l0 * patch_n_h_l0)

    # return(thumbnail as array, patch_n_w_l0, patch_n_h_l0)
    return (p_s, patch_n_w_l0, patch_n_h_l0, mpp, w_l0, h_l0, obj_power)
