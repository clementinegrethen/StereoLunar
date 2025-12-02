from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import FigureBase


class DisplayVideo:
    """
    Display Video

    Utility for displaying consecutive images on the same matplotlib plot
    """

    def __init__(self, figure_name: str = "figure"):
        self.fig: Optional[FigureBase] = None
        self.ax_im: Optional[Axis] = None
        self.ax_d: Optional[Axis] = None
        self.im_widget: Any = None
        self.d_widget: Any = None
        self.figure_name: str = figure_name

    def __call__(
        self, float_image: npt.NDArray[np.float32], dmap: npt.NDArray[np.float32]
    ):
        if self.ax_im is None:
            self.fig, (self.ax_im, self.ax_d) = plt.subplots(2, 1, num=self.figure_name)
            self.fig.suptitle(self.figure_name)
            if float_image is not None:
                self.im_widget = self.ax_im.matshow(
                    float_image, cmap="gray", interpolation="none"
                )
            self.d_widget = self.ax_d.matshow(dmap, interpolation="none")
            plt.colorbar(self.d_widget)

            self.ax_im.axis("off")
            self.ax_d.axis("off")

        else:
            if float_image is not None:
                self.im_widget.set_data(float_image)
            self.d_widget.set_data(dmap)

        plt.draw()
        plt.pause(0.001)
