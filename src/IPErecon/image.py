"""Module containing the Image class. Image class takes care of loading and processing of
images.
"""

import re
import cv2
import pytesseract
import numpy as np
import bm3d
import matplotlib.pyplot as plt
from IPErecon.helpers import *

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
# https://github.com/UB-Mannheim/tesseract/wiki


class Image:
    data = (
        []
    )  # results of surface coverage are saved here when save_surface() is called
    speed = (
        []
    )  # results of speed of growth are saved here when speed_calculation() is called
    roi = None

    def __init__(self, image: np.ndarray, name: str):
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array.")
        self.images = {"image": image}
        self.metadata = {"name": name, "units": None, "pixel_size": None}
        self.results = {"surface_coverage": None, "surface_percentage": None}

    @property
    def image(self):
        return self.images.get("image", None)

    @classmethod
    def load_image(cls, path: str, name: str):
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cls(image, name)

    @staticmethod
    def _scale_calculation(scale_bar):
        """
        Function for scale calculation. Uses scale bar that was created by cropp() function
        It saves length of reference line in pixels, scale numerical value and units

        For tif images, maybe can be added to recognise those values from tif metadata
        """

        # calculation of length of the reference line
        a = np.zeros(scale_bar.shape[0])
        for i in range(scale_bar.shape[1] - 1):
            for j in range(scale_bar.shape[0] - 3):
                if (
                    scale_bar[j][i] == 255
                    and scale_bar[j + 1][i] == 255
                    and scale_bar[j + 2][i] == 255
                    and scale_bar[j + 3][i] == 255
                    and scale_bar[j][i + 1] == 255
                    and scale_bar[j + 1][i + 1] == 255
                    and scale_bar[j + 2][i + 1] == 255
                    and scale_bar[j + 3][i + 1] == 255
                ):
                    a[j] += 1

        length = max(a)

        # new size for the scale bar - better for character recognition
        width2 = scale_bar.shape[1] * 3
        height2 = scale_bar.shape[0] * 3
        dim = (width2, height2)

        # resize of scale_bar for better recognition
        scale_bar_big = cv2.resize(scale_bar, dim, interpolation=cv2.INTER_AREA)
        # algoryth for character recognition
        text = pytesseract.image_to_string(scale_bar_big, config="--psm 6")
        text = re.sub("[^A-Za-z0-9]+", "", text)
        scaleunits = [
            ["cm", "mm", "um", "nm", "pm"],
            [10 ** (-2), 10 ** (-3), 10 ** (-6), 10 ** (-9), 10 ** (-12)],
        ]
        scalenum = int(re.search(r"\d+", text).group())
        units = str(re.search("[a-z][a-z]", text).group())
        units2 = units

        # it is unused - maybe can be deleted
        for i in range(np.shape(scaleunits)[1]):
            if units == scaleunits[0][i]:
                scalevalue = scalenum * scaleunits[1][i]
        #

        # it is unable to recognise micrometers - this fixes the problem
        if units2 == "um":
            units2 = "\mu m"

        return units2, scalevalue / length

    def cropp_legend(
        self, calculate_scale=True
    ):  # revrites image to cropped image, makes scale bar
        """
        Image is separeted into legend and SEM image.
         From legend, scale bar is cropped.
        Both scale bar and SEM image is saved.
        Could be upgraded for both Thermofisher and Tescan images.
        """

        height, width = self.image.shape
        # crop of legend
        legend = self.image[width:height, 0:width]
        self.images["image"] = self.image[0:width, 0:width]
        height1, width1 = legend.shape

        left = int(3.33 / 5 * width1)
        top = 0
        right = width1
        bottom = int(height1 / 2)

        # crop of scale bar
        scale_bar = legend[top:bottom, left:right]
        if calculate_scale:
            units, pixel_size = self._scale_calculation(scale_bar)
            self.metadata["units"] = units
            self.metadata["pixel_size"] = pixel_size

    def define_axes(self, units: str, pixel_size: float):
        self.metadata["units"] = units
        self.metadata["pixel_size"] = pixel_size

    def cropp(self):
        if not self.roi:
            new_roi = CropImage(self.image)
            Image.roi = new_roi.roi
            self.images["image"] = self.images["image"][
                Image.roi[2] : Image.roi[3], Image.roi[0] : Image.roi[1]
            ]
        else:
            self.images["image"] = self.images["image"][
                Image.roi[2] : Image.roi[3], Image.roi[0] : Image.roi[1]
            ]

    def _nlmean_denoising(self, **kwargs):
        defaultkwargs = {"h": 40, "templateWindowSize": 7, "searchWindowSize": 21}
        defaultkwargs.update(kwargs)
        self.images["image"] = cv2.fastNlMeansDenoising(
            self.images["image"], **defaultkwargs
        )

    def _bm3d_denoising(self, **kwargs):
        defaultkwargs = {"sigma_psd": 40, "stage_arg": bm3d.BM3DStages.ALL_STAGES}
        defaultkwargs.update(kwargs)
        self.images["image"] = bm3d.bm3d(self.image, **defaultkwargs).astype(np.uint8)

    def _median_denoising(self, **kwargs):
        defaultkwargs = {"ksize": 5}
        defaultkwargs.update(kwargs)
        self.images["image"] = cv2.medianBlur(self.image, **defaultkwargs)
