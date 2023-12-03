import time

import ipdb
import numpy as np
import Xlib.display
import Xlib.X as X
from loguru import logger
from PIL import Image
from tiny_taxinet import getCurrentImage


def get_image():
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
    finally:
        display.close()


def main():
    # BGRA
    # (1080, 1920, 4)
    image_arr = get_image()

    # BGRA -> RGB.
    image_arr = image_arr[:, :, [2, 1, 0]]

    # Save the image.
    im = Image.fromarray(image_arr, mode="RGB")
    im.save("test_screenshot.png")

    # Compare with the existing screenshot.
    logger.info("Comparing with the existing screenshot in 2s...")
    time.sleep(2)
    getCurrentImage()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
