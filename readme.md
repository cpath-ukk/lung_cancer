Reference of provided code

NB! We provide raw working scripts used for the development and implementation.
Upon publication we will annotate and prepare code for end users.

NB! Most of the experiments and scripts were run in anaconda environment
using Ubuntu 22.04. We recommend using this setup.

## 01_qupath_patch_extraction
This a Groovy script for use with QuPath v. > 0.3 for extraction of training
or validation/test patches and associated segmentation masks (ground truth
from annotations).

## 02_train_pixel_wise_segmentation
train_script.py: This is a training script.
It was used with the Python v.3.9
and pytorch v.1.10.
Segmentation-models-pytorch v.0.3.1 library (or any other version) is
used for model construction.
pytorch_toolbelt is used for construction of some of the loss functions (v.0.6.3)

dataset_v2.py: Helper script for dataset construction and data augmentation.
Albumentations package (v1.3.0) was used for data augmentation.

## 03_Patch_level_validation_testing
This is a Python script for additional validation or test of the trained checkpoints
using patch-level metrics of segmentation accuracy (Dice score and IoU).

## 04_WSI_inference
This is an inference script to process the whole-slide images (WSI).

Three versions are available: 1) for main segmentation algorithm (uses
slides and tissue detection masks as input), 2) for subtyping/supervised pixel-wise
segmentation algorithm (uses multi-class tissue segmentation mask from main
algorithm as input + slides), 3) same as 2 but for classifiers based on
UNI and Prov-GigaPath feature extractors (original model checkpoints from
hugging face and trained classifiers upon features are necessary).

NB! We provide the script for review that is based on the openslide (v.3.4.1
or later) library.

-wsi_tis_detect.py: script to run isolated tissue vs background segmentation
(based on the segmentation algorithm from GrandQC tool - manuscript in review,
the rool will be open-sourced upon publications). For custom tissue detectors,
this script can be ignored and tissue detection map can be used in main.py from
other sources.
-main.py: script to start inference using a trained pixel-wise segmentation model.
Needs tissue mask as input that should be stored in outputs folder.

-run.sh: bash script to pipe the tissue segmentation and artifact detection
steps (can be ignored by custom tissue detection pipelines)

-wsi_colors.py: script where color scheme to be used is defined
-wsi_maps.py: script with function to make overlay of segmentation
mask on the original WSI
-wsi_process.py: script with processing pipeline of WSI, output:
segmentation mask (class codings), segmentation mask (RGB corresponding to
wsi_colors.py defined scheme)
-wsi_slide_info.py: pipeline to retrieve WSI metadata necessary for inference.
-wsi_tis_detect_helper_fx.py: helper functions for tissue segmentation script
(wsi_tis_detect.py)

The output txt file with processed slides will be saved.
For subtyping versions it will include subtype areas and final slide classification.

## 05_foundation_model_classifier_training
Scripts for training a supervised classifier for lung cancer subtyping
based on features extracted by foundation models.

-prepare_ds_prov_gigapath_gpu_monodir.py (UNI foundation model)
-prepare_ds_uni_gpu_multidir.py (Prov-GigaPath foundation model)
Scripts for preparing datasets for training/validation/test, effectively,
for transfering image patches into feature vectors that will be an input
for training script.

-train_validation.py
Training script that trains a fully connected layer and a classification layer
upon the extracted features.

## 06_foundation_model_classifier_test_patch_level
Test script for patch-level accuracy evaluation of classifiers that
are based on the foundation models (from ##5 step above).
Output: "probability" of subtype at patch-level




