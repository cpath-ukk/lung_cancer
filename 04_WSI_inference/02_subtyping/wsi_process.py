# MAIN LOOP TO PROCESS WSI
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
from tqdm import tqdm

#Helper functions
def to_tensor_x(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(image, preprocessing_fn, model_size):
    if image.size != model_size:
        image = image.resize(model_size)
        print('resized')
    image = np.array(image)
    x = preprocessing_fn(image)
    x = to_tensor_x(x)
    return x

def make_1class_map_thr (mask, class_colors):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(1, len(class_colors)+1):
        idx = mask == l
        r[idx] = class_colors [l-1][0]
        g[idx] = class_colors [l-1][1]
        b[idx] = class_colors [l-1][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def slide_process_single(model, tis_det_map_mpp, slide, patch_n_w_l0, patch_n_h_l0, p_s, m_p_s, colors,
                         ENCODER_MODEL_1,ENCODER_WEIGHTS, DEVICE, BACK_CLASS):
    '''
    Tissue detection map is generated under MPP = 4, therefore model patch size of (512,512) corresponds to tis_det_map patch
    size of (128,128).
    '''

    model_size = (m_p_s, m_p_s)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER_MODEL_1, ENCODER_WEIGHTS)

    # Start loop
    for he in tqdm(range(patch_n_h_l0), total=patch_n_h_l0):
        h = he * p_s + 1
        if (he == 0):
            h = 0
        #print("Current cycle ", he + 1, " of ", patch_n_h_l0)
        for wi in range(patch_n_w_l0):
            w = wi * p_s + 1
            if (wi == 0):
                w = 0
            #he = 12
            #wi = 15
            td_patch = tis_det_map_mpp [he*m_p_s:(he+1)*m_p_s,wi*m_p_s:(wi+1)*m_p_s]
            if td_patch.shape != (512,512):
                # td_patch padding (incase td_patch does not equal (512,512))
                original_shape = td_patch.shape

                # Desired shape
                desired_shape = (512, 512)

                # Calculate padding needed
                padding = [(0, desired_shape[i] - original_shape[i]) for i in range(2)]

                # Apply padding
                td_patch_ = np.pad(td_patch, padding, mode='constant')
            else:
                td_patch_ = td_patch

            if np.count_nonzero(td_patch_ == 1) > 50: #here change to check of segmentation map (== 1 are tumor pixels)
                # Generate patch
                work_patch = slide.read_region((w, h), 0, (p_s, p_s))
                work_patch = work_patch.convert('RGB')

                # Resize to model patch size
                work_patch = work_patch.resize((m_p_s, m_p_s), Image.ANTIALIAS)

                image_pre = get_preprocessing(work_patch, preprocessing_fn, model_size)
                x_tensor = torch.from_numpy(image_pre).to(DEVICE).unsqueeze(0)
                predictions = model.predict(x_tensor)
                predictions = (predictions.squeeze().cpu().numpy())

                mask_raw = np.argmax(predictions, axis=0).astype('int8')
                mask = np.where(mask_raw == 1, 13, td_patch_)
                mask = np.where(mask_raw == 2, 14, mask)

                #mask_1class = make_1class_map_thr(mask, colors)
            else:
                mask = td_patch_


            if (wi == 0):
                temp_image = mask

            else:
                temp_image = np.concatenate((temp_image, mask), axis=1)

        if (he == 0):
            end_image = temp_image

        else:
            end_image = np.concatenate((end_image, temp_image), axis=0)


    end_image_1class = make_1class_map_thr(end_image, colors)
    end_image_1class = Image.fromarray(end_image_1class)
    end_image_1class = end_image_1class.resize((patch_n_w_l0*50, patch_n_h_l0*50), Image.ANTIALIAS)

    return (end_image_1class, end_image)