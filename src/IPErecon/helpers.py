import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np

def normal_round(number: float):
    """Round a float to the nearest integer."""
    return int(number + 0.5)

class FixedSizeRectangleSelector(RectangleSelector):
    
    def _onmove(self, event):
        # Start bbox
        s_x0, s_x1, s_y0, s_y1 = self.extents
        start_width = np.float16(s_x1 - s_x0)
        start_height = np.float16(s_y1 - s_y0)
        
        super()._onmove(event)
        
        # Custom behavior only if selector is moving
        if not self._active_handle == 'C':  
            return
        
        # End bbox
        e_x0, e_x1, e_y0, e_y1 = self.extents
        end_width = np.float16(e_x1 - e_x0)
        end_height = np.float16(e_y1 - e_y0)

        if start_width != end_width:
            e_x0, e_x1 = s_x0, s_x1
            
        if start_height != end_height:
            e_y0, e_y1 = s_y0, s_y1
            
        self.extents = e_x0, e_x1, e_y0, e_y1

class CropImage:

    def __init__(self, image: np.ndarray):
        self.image = image
        self.figure, self.axis = None, None
        self.roi = None

        self._init_plot()

    def _init_plot(self):
        self.figure , self.axis = plt.subplots()
        self.axis.imshow(self.image, cmap="gray")
        self.toggle_selector = FixedSizeRectangleSelector(self.axis, self._select_callback,
                                                          useblit=True,
                                                          button=[1],  # don't use middle button
                                                          minspanx=5, minspany=5,
                                                          spancoords='pixels',
                                                          interactive=True,
                                                          drag_from_anywhere=True)
        self.figure.canvas.mpl_connect("close_event", self._close_event)
        self.figure.canvas.draw()
        self.figure.suptitle(
            "To select square in second image, press space. Press enter to confirm."
        )
        plt.show()

    def _select_callback(self, eclick, erelease):
        """Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata


    def _close_event(self, event):
        self.roi = np.array(
                [normal_round(x) for x in self.toggle_selector.extents])