"""Module containing the Image class. Image class takes care of loading and processing of
images.
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import color

class Image:
    data = [] # results of surface coverage are saved here when save_surface() is called
    speed = [] # results of speed of growth are saved here when speed_calculation() is called
    roi = None

    def __init__(self, image: np.ndarray, name: str):
        self.images = {'image': image}
        self.metadata = {'name': name}
        self.results = {}
    
    @property
    def image(self):
        return self.images.get("image", None)
    
    @classmethod
    def load_image(cls, path: str,name: str):
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")
        image = io.imread(path,as_gray=True)
        return cls(image, name)
        