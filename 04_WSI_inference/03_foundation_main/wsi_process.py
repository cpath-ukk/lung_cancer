# MAIN LOOP TO PROCESS WSI
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

#Helper functions
def get_embedding(model, image, device, transform):
    sample_input = transform(image).unsqueeze(0).to(device)  # Move input to GPU
    with torch.no_grad():
        output = model(sample_input).squeeze()
    return output

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


def slide_process_single(model_prim, tis_det_map_mpp, slide, patch_n_w_l0, patch_n_h_l0, p_s, m_p_s, colors,
                         DEVICE, BACK_CLASS, model_clas, THR, transform):
    '''
    Tissue detection map is generated under MPP = 4, therefore model patch size of (512,512) corresponds to tis_det_map patch
    size of (128,128).
    '''

    model_size = (m_p_s, m_p_s)

    model_prim.to(DEVICE)

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
            if td_patch.shape != (m_p_s,m_p_s):
                # td_patch padding (incase td_patch does not equal (512,512))
                original_shape = td_patch.shape

                # Desired shape
                desired_shape = (m_p_s,m_p_s)

                # Calculate padding needed
                padding = [(0, desired_shape[i] - original_shape[i]) for i in range(2)]

                # Apply padding
                td_patch_ = np.pad(td_patch, padding, mode='constant')
            else:
                td_patch_ = td_patch

            if np.count_nonzero(td_patch_ == 1) > 10: #here change to check of segmentation map (== 1 are tumor pixels)
                # Generate patch
                work_patch = slide.read_region((w, h), 0, (p_s, p_s))
                work_patch = work_patch.convert('RGB')

                # Resize to model patch size
                work_patch = work_patch.resize((m_p_s, m_p_s), Image.ANTIALIAS)

                embedding = get_embedding(model_prim, work_patch, DEVICE, transform)
                outp = model_clas(embedding).squeeze()
                outp_np = outp.cpu().detach().numpy() # 0-1 for probability

                if outp_np < THR:
                    mask = np.where(td_patch_ == 1, 13, td_patch_) # LUAD
                else:
                    mask = np.where(td_patch_ == 1, 14, td_patch_)  # LUSC

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