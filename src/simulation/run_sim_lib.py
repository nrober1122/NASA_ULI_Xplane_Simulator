import os
import pathlib
import sys
import time
from typing import Callable, NamedTuple

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from xplane_screenshot import get_xplane_image

import xpc3
import xpc3_helper
from simulation.static_atk import StaticAttack
from simulation.tiny_taxinet2 import StateEstimator


class SimResult(NamedTuple):
    T_t: np.ndarray
    T_state_gt: np.ndarray
    T_state_clean: np.ndarray
    T_state_est: np.ndarray

    T_image_raw: np.ndarray
    T_image_clean: np.ndarray
    T_image_est: np.ndarray


def get_state(
    image_raw: np.ndarray, attack: StaticAttack, est_state, linfnorm: float = None, should_attack: bool = True
):
    # Process image before passing it to NN estimator
    image_processed = attack.process_image(image_raw)

    # Add adversarial attack if applicable
    if should_attack:
        image_processed += attack.get_patch(image_processed, linfnorm)
        image_processed = image_processed.clip(0, 1)

    # Estimate state from processed image
    cte, he = est_state(image_processed)

    return cte, he, image_processed


def simulate_controller(
    client: xpc3.XPlaneConnect,
    attack: StaticAttack,
    linfnorm: float,
    startCTE: float,
    startHE: float,
    startDTP: float,
    endDTP: float,
    get_control: Callable,
    dt: float = 0.1,
    sim_speed: float = 1.0,
) -> SimResult:
    client.pauseSim(True)

    client.sendDREF("sim/time/sim_speed", sim_speed)
    xpc3_helper.reset(client, cteInit=startCTE, heInit=startHE, dtpInit=startDTP)
    xpc3_helper.sendBrake(client, 0)
    time.sleep(0.05)
    client.pauseSim(False)

    dtp = startDTP
    startTime = client.getDREF("sim/time/zulu_time_sec")[0]
    now = startTime
    endTime = startTime
    run_end_time = now + 30.0

    T_state_gt, T_state_clean, T_state_est = [], [], []
    T_image_raw = []
    T_image_clean, T_image_est = [], []
    T_rudder_clean = []
    T_rudder = []

    cte_gt, dtp_gt, he_gt = xpc3_helper.getHomeState(client)

    est_state = StateEstimator(stride=attack.stride)

    while dtp < endDTP and now < run_end_time and np.abs(cte_gt) < 10.5:
        speed = xpc3_helper.getSpeed(client)
        throttle = 0.1
        # Note: Why bang-bang controller?
        if speed > 5:
            throttle = 0.0
        elif speed < 3:
            throttle = 0.2

        cte_gt, dtp_gt, he_gt = xpc3_helper.getHomeState(client)
        state_gt = np.array([cte_gt, dtp_gt, he_gt])
        T_state_gt.append(state_gt)

        image_raw = get_xplane_image()
        T_image_raw.append(image_raw)

        cte, he, img = get_state(image_raw, attack, linfnorm=linfnorm, should_attack=True)
        rudder = get_control(client, cte, he)
        client.sendCTRL([0, rudder, rudder, throttle])

        logger.info("CTE: {: .2f} ({: .2f}), HE: {: .2f} ({: .2f}), RU: {: .2f}".format(cte, cte_gt, he, he_gt, rudder))

        cte_clean, he_clean, img_clean = get_state(image_raw, attack, est_state, should_attack=False)
        rudder_clean = get_control(client, cte_clean, he_clean)
        T_rudder_clean.append(rudder_clean)
        T_rudder.append(rudder)

        T_image_clean.append(img_clean)
        T_image_est.append(img)

        state_est = np.array([cte, dtp_gt, he])
        T_state_est.append(state_est)
        state_clean = np.array([cte_clean, dtp_gt, he_clean])
        T_state_clean.append(state_clean)

        # Wait for next timestep. 1 Hz control rate?
        while endTime - startTime < dt:
            endTime = client.getDREF("sim/time/zulu_time_sec")[0]
            time.sleep(0.001)

        # Set things for next round
        startTime = client.getDREF("sim/time/zulu_time_sec")[0]
        endTime = startTime
        _, dtp, _ = xpc3_helper.getHomeState(client)
        time.sleep(0.001)
        now = startTime

    client.pauseSim(True)

    T_t = np.arange(len(T_state_gt)) * dt

    T_state_gt = np.stack(T_state_gt, axis=0)
    T_state_clean = np.stack(T_state_clean, axis=0)
    T_state_est = np.stack(T_state_est, axis=0)

    T_image_raw = np.stack(T_image_raw, axis=0)
    T_image_clean = np.stack(T_image_clean, axis=0)
    T_image_est = np.stack(T_image_est, axis=0)

    return SimResult(T_t, T_state_gt, T_state_clean, T_state_est, T_image_raw, T_image_clean, T_image_est)
