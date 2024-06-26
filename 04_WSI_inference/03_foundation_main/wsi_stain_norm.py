
import numpy as np
import cv2 as cv

def is_image(x):
    """
    Is x an image?
    i.e. numpy array of 2 or 3 dimensions.

    :param x: Input.
    :return: True/False.
    """
    if not isinstance(x, np.ndarray):
        return False
    if x.ndim not in [2, 3]:
        return False
    return True


def is_uint8_image(x):
    """
    Is x a uint8 image?

    :param x: Input.
    :return: True/False.
    """
    if not is_image(x):
        return False
    if x.dtype != np.uint8:
        return False
    return True

class BrightnessStandardizer(object):
    """
    A class for standardizing image brightness. This can improve performance of other normalizers etc.
    """

    def __init__(self):
        pass

    def transform(self, I, percentile=95):
        """
        Transform image I to standard brightness.
        Modifies the luminosity channel such that a fixed percentile is saturated.

        :param I: Image uint8 RGB.
        :param percentile: Percentile for luminosity saturation. At least (100 - percentile)% of pixels should be fully luminous (white).
        :return: Image uint8 RGB with standardized brightness.
        """
        assert is_uint8_image(I), "Image should be RGB uint8."
        I_LAB = cv.cvtColor(I, cv.COLOR_RGB2LAB)
        L_float = I_LAB[:, :, 0].astype(float)
        p = np.percentile(L_float, percentile)
        I_LAB[:, :, 0] = np.clip(255 * L_float / p, 0, 255).astype(np.uint8)
        I = cv.cvtColor(I_LAB, cv.COLOR_LAB2RGB)
        return I


#Inititate BrightnessStandardizer
standardizer = BrightnessStandardizer()
