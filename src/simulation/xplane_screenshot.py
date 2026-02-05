import time

import cv2
import einops as ei
import ipdb
import numpy as np
import Xlib.display
import Xlib.X as X
from loguru import logger
from PIL import Image
from simulators.NASA_ULI_Xplane_Simulator.src.simulation.tiny_taxinet import getCurrentImage


def get_xplane_image() -> np.ndarray:
    """ Returns an image of the current X-Plane 11 window. (1080, 1920, 4).
    """
    display = Xlib.display.Display()

    root = display.screen().root
    windowIDs = root.get_full_property(display.intern_atom("_NET_CLIENT_LIST"), X.AnyPropertyType).value

    try:
        for windowID in windowIDs:
            window = display.create_resource_object("window", windowID)
            window_title_property = window.get_full_property(display.intern_atom("_NET_WM_NAME"), 0)
            window_title = window_title_property.value.decode("utf-8")

            if window_title == "X-System":
                geometry = window.get_geometry()
                width, height = geometry.width, geometry.height
                pixmap = window.get_image(0, 0, width, height, X.ZPixmap, 0xFFFFFFFF)
                data = pixmap.data
                final_image = np.frombuffer(data, dtype="uint8").reshape((height, width, 4))
                return final_image

        raise Exception("X-Plane 11 window not found!")
    finally:
        display.close()
