from typing import NamedTuple

import numpy as np


class SimResult(NamedTuple):
    T_t: np.ndarray
    # (T, 3). [cte, dtp, he]
    T_state_gt: np.ndarray
    T_state_clean: np.ndarray
    T_state_est: np.ndarray

    T_image_raw: np.ndarray
    T_image_clean: np.ndarray
    T_image_est: np.ndarray

    def without_images(self):
        return self._replace(T_image_raw=None, T_image_clean=None, T_image_est=None)
