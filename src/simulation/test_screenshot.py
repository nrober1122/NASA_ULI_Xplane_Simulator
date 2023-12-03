import time

import cv2
import einops as ei
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


def process_image(image_arr: np.ndarray):
    assert image_arr.shape == (1080, 1920, 4)
    # 0: Remove alpha channel.
    image = image_arr[:, :, :3]

    # 1: Crop 100 pixels from the top and bottom, 100 pixels from the left, 20 pixels from the right.
    image = image[100:-20, 100:-100, :]
    assert image.shape == (960, 1720, 3)

    # 2: Crop 230 more pixels from the top.
    image = image[230:, :, :]
    assert image.shape == (730, 1720, 3)

    # 3: Resize image to 360x200.
    image = cv2.resize(image, (360, 200))
    assert image.shape == (200, 360, 3)

    # 4: BGR -> RGB.
    image = image[:, :, ::-1]

    # 5: Convert to grayscale.
    image = Image.fromarray(image).convert("L")

    # 6: Crop out nose, sky, bottom of image.
    image = image.crop((55, 5, 360, 135)).resize((256, 128))
    image = np.array(image) / 255.0
    assert image.shape == (128, 256)

    return image


def downsample_image(image: np.ndarray):
    assert image.shape == (128, 256)
    height_orig, width_orig = image.shape

    stride = 16
    num_pix = 16
    width = width_orig // stride
    height = height_orig // stride

    # out = np.zeros((height, width))
    # for ii in range(height):
    #     for jj in range(width):
    #         patch = image[stride * ii : stride * (ii + 1), stride * jj : stride * (jj + 1)]
    #         brightest_pixels = np.sort(patch.flatten())[-num_pix:]
    #         out[ii, jj] = np.mean(brightest_pixels)

    # Do the same computation as above but vectorized.
    patches = ei.rearrange(image, "(nh sh) (nw sw) -> nh nw sh sw", nw=width, nh=height, sw=stride, sh=stride)
    patches_flat = ei.rearrange(patches, "nh nw sh sw -> nh nw (sh sw)")
    out2 = np.mean(np.sort(patches_flat, axis=-1)[:, :, -num_pix:], axis=-1)

    # assert np.allclose(out, out2)
    return out2


def main():
    # BGRA
    # (1080, 1920, 4)
    image_arr = get_image()

    image_downsampled = downsample_image(process_image(image_arr))
    Image.fromarray((image_downsampled * 255).astype(np.uint8)).save("screenshot_downsampled.png")

    # Save the image.
    im = Image.fromarray(image_arr[:, :, [2, 1, 0]], mode="RGB")
    im.save("test_screenshot.png")

    # Compare with the existing screenshot.
    logger.info("Comparing with the existing screenshot in 2s...")
    time.sleep(2)
    getCurrentImage()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
