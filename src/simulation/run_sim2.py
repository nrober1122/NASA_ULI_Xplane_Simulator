import os
import sys
from datetime import datetime

import pypalettes

xpc3_dir = os.environ["NASA_ULI_ROOT_DIR"] + "/src/"
results_dir = os.environ["NASA_ULI_DATA_DIR"] + "/logs/LOG_" + datetime.now().strftime("%Y%m%d_%H_%M_%S") + "/"


sys.path.append(xpc3_dir)

import time
from typing import Callable

import ipdb
import time
import matplotlib.pyplot as plt
import numpy as np
from simulators.NASA_ULI_Xplane_Simulator.src.simulation import settings
# import static_atk
# import tiny_taxinet
# import tiny_taxinet2
from loguru import logger
# from tiny_taxinet import process_image
from simulators.NASA_ULI_Xplane_Simulator.src.simulation.xplane_screenshot import get_xplane_image
from PIL import Image
import jax.numpy as jnp
import jax

import xpc3
import xpc3_helper

from utils.torch2jax import torch2jax
import hj_reachability as hj
from hjnnv import hjnnvUncertaintyAwareFilter
from utils.attacks import fgsm, pgd
import dynamic_models
import simulators.NASA_ULI_Xplane_Simulator.src.simulation.tiny_taxinet2
import json
import pickle

import matplotlib.pyplot as plt

jax.config.update('jax_platform_name', 'cpu')


def get_state(
        image_raw: np.ndarray,
        x_prev: jnp.ndarray = None,
        u_prev: jnp.ndarray = None,
        attack: Callable = None,
        target: jnp.ndarray = None,
        adv_ptb_prev=None,
        key=None
        ):
    # Process image before passing it to NN estimator
    image_raw_noisy = np.copy(image_raw)
    if key is not None and False:
        image_raw_jnp = jnp.array(image_raw)
        noise = np.array(jax.random.uniform(key, shape=image_raw_jnp.shape, minval=-settings.IMAGE_NOISE, maxval=settings.IMAGE_NOISE) * 255, dtype='int')
        # ipdb.set_trace()
        image_raw = np.clip(image_raw + noise, 0, 255).astype('uint8')
    # ipdb.set_trace()
    image_processed = settings.PROCESS_IMG(image_raw)
    image_processed_jnp = jnp.array(image_processed)
    if key is not None:
        image_processed_jnp = jnp.array(image_processed)
        noise = jax.random.uniform(key, shape=image_processed_jnp.shape, minval=-settings.IMAGE_NOISE, maxval=settings.IMAGE_NOISE)
        image_processed = np.clip(image_processed + np.array(noise), 0.0, 1.0)

    if not settings.USING_TORCH:
        image_processed = jax.numpy.array(image_processed)

    # Add adversarial attack if applicable
    if 'static' in settings.ATTACK_STRING:
        # image_processed += attack.get_patch(image_processed, 0.032)
        if attack is not None:
            image_processed += settings.ATTACK.get_patch(image_processed, settings.ATTACK_STRENGTH)
        # pil_img = Image.fromarray((attack.get_patch(image_processed).reshape((8,16)))*255+125)
        # pil_img = pil_img.convert("L")
        # pil_img.save(results_dir + 'mask.png')

        # pil2 = Image.fromarray((attack.get_patch(image_processed, 0.022).transpose([1, 2, 0]) * 225).astype(np.uint8))
        # pil2.save(results_dir + 'dnn_mask.png')

        # import pdb; pdb.set_trace()
    elif settings.ATTACK_STRING == 'fgsm':
        print("Using FGSM attack")
        if target is None and attack is not None:
            raise ValueError("Target must be provided for FGSM attack")
        elif attack is not None:
            adv_image, loss, perturbation = fgsm(
                model=settings.GET_STATE,
                target=jnp.array(target),
                observation=jnp.array(image_processed),
                epsilon=settings.ATTACK_STRENGTH,
                loss_fn=attack,
                key=key,
                perturbation_prev=adv_ptb_prev
            )
            if adv_ptb_prev is not None and loss > 100.0:
                print("Reusing previous adv image")
                perturbation = adv_ptb_prev
            image_processed += perturbation
            print(f"FGSM attack loss: {loss}")
    elif settings.ATTACK_STRING == 'pgd':
        print("Using PGD attack")
        if target is None and attack is not None:
            raise ValueError("Target must be provided for PGD attack")
        elif attack is not None:
            print("Using PGD attack")
            adv_image, loss, perturbation = pgd(
                model=settings.GET_STATE,
                target=jnp.array(target),
                observation=jnp.array(image_processed),
                epsilon=settings.ATTACK_STRENGTH,
                iters=settings.PGD_STEPS,
                alpha=settings.PGD_ALPHA,
                loss_fn=attack,
                key=key,
                perturbation_prev=adv_ptb_prev
            )
            if adv_ptb_prev is not None and loss > 100.0:
                perturbation = adv_ptb_prev
            image_processed += perturbation
            print(f"PGD attack loss: {loss}")

    elif settings.ATTACK_STRING == 'null':
        pass

    elif settings.ATTACK_STRING is not None:
        raise ValueError("Attack type not recognized")

    # Estimate state from processed image

    if settings.SMOOTHING and x_prev is not None:
        observation = settings.PACKAGE_INPUT(x_prev, u_prev, image_processed)
        # observation = jnp.concatenate([x_prev, u_prev, image_processed])
        cte, he = settings.GET_STATE_SMOOTHED(observation)
    else:
        observation = image_processed
        cte, he = settings.GET_STATE(observation)

    image_attacked_jnp = jnp.array(image_processed)
    if attack is not None and settings.ATTACK_STRING != 'null':
        print("Max perturbation end of fcn:", jnp.max(jnp.abs(image_attacked_jnp - image_processed_jnp)))

    return cte, he, image_processed, perturbation if 'perturbation' in locals() else None


def main():
    with xpc3.XPlaneConnect() as client:
        # Set weather and time of day
        client.sendDREF("sim/time/zulu_time_sec", settings.TIME_OF_DAY * 3600 + 8 * 3600)
        client.sendDREF("sim/weather/cloud_type[0]", settings.CLOUD_COVER)

        if settings.DUBINS:
            simulate_controller_dubins(
                client,
                settings.START_CTE,
                settings.START_HE,
                settings.START_DTP,
                settings.END_DTP,
                get_state,
                settings.GET_CONTROL,
                settings.DT,
                settings.CTRL_EVERY,
            )
        else:
            simulate_controller(
                client,
                settings.START_CTE,
                settings.START_HE,
                settings.START_DTP,
                settings.END_DTP,
                get_state,
                settings.GET_CONTROL,
            )


def simulate_controller(
    client: xpc3.XPlaneConnect,
    startCTE: float,
    startHE: float,
    startDTP: float,
    endDTP: float,
    get_state: Callable,
    get_control: Callable,
    sim_speed: float = 1.0,
):
    client.pauseSim(True)

    client.sendDREF("sim/time/sim_speed", sim_speed)
    xpc3_helper.reset(client, cteInit=startCTE, heInit=startHE, dtpInit=startDTP)
    xpc3_helper.sendBrake(client, 0)

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            np.array([-15., -np.pi/4]),
            np.array([15., np.pi/4])),
        (100, 100),
    )
    values = -jnp.abs(grid.states[..., 0]) + 10.0

    # Set up the uncertainty-aware filter
    hjnnv_filter = hjnnvUncertaintyAwareFilter(
        dynamic_models.TaxiNetDynamics(),
        pred_model=settings.NETWORK(),
        grid=grid,
        initial_values=values,
        num_controls=50,
        num_disturbances=30,
    )

    # Use random calculation to get jit compilation before the main loop
    v_star, u_star, worst_val, val_filter, _, _ = hjnnv_filter.ua_filter(
            jnp.array(np.tan(np.deg2rad(0))),
            hj.sets.Box(
                jnp.array([-0.1, -0.1]),
                jnp.array([0.1, 0.1])
            ),
            num_states=30,
    )

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

    dt = settings.DT

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

        # Get the estimated state and control without attack
        cte_clean, he_clean, img_clean = get_state(image_raw)
        rudder_clean = get_control(client, cte_clean, he_clean)

        # Get the estimated state and control with attack
        cte, he, img = get_state(image_raw, attack=settings.ATTACK)
        logger.info("CTE: {:.2f} ({:.2f}), HE: {:.2f} ({:.2f})".format(cte, cte_gt, he, he_gt))
        rudder = get_control(client, cte, he)
        # ipdb.set_trace()

        # Use the uncertainty-aware filter to get the control input
        state_bounds = hjnnv_filter.nnv_state_bounds(
            jnp.array([cte, np.deg2rad(he)]),
            jnp.abs(jnp.array([cte-cte_gt, np.deg2rad(he-he_gt)]))
        )
        # ipdb.set_trace()
        v_star, u_star, worst_val, val_filter = hjnnv_filter.ua_filter(
            jnp.array(np.tan(np.deg2rad(rudder))),
            state_bounds,
            num_states=30,
        )
        print("CTE error: {:.2f}, HE error: {:.2f}".format(cte-cte_gt, he-he_gt))
        print("Best value: {:.2f}, Best control: {:.2f}, Worst value: {:.2f}, Filtered value: {:.2f}".format(
            v_star, u_star, worst_val, val_filter
        ))
        print(f"Control: {rudder}, Filtered Control: {np.rad2deg(np.arctan(u_star))}")
        client.sendCTRL([0, rudder, rudder, throttle])

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

    labels = ["CTE (m)", "DTP (m)", "HE (degrees)"]
    # Plot.
    fig, axes = plt.subplots(3, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_state_gt[:, ii], color="C4", label="True")
        ax.plot(T_t, T_state_est[:, ii], color="C1", label="Estimated")
        ax.set_ylabel(labels[ii], rotation=0, ha="right")
    axes[0].legend()
    fig.savefig(results_dir + "sim2_traj.pdf")
    plt.close(fig)
    ###############################################3
    # x axis = DTP, y axis = CTE.
    cte_constr = 10.0
    ylim = 11.0
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_state_gt[:, 1], T_state_gt[:, 0], marker="o", lw=0.5, color="C1")
    ax.set_aspect("equal")
    ymin, ymax = ax.get_ylim()
    ymin, ymax = min(ymin, -ylim), max(ymax, ylim)
    ax.set_ylim(ymin, ymax)
    ax.axhspan(cte_constr, ymax, color="C0", alpha=0.2)
    ax.axhspan(-cte_constr, ymin, color="C0", alpha=0.2)
    fig.savefig(results_dir + "sim2_plot2d.pdf")
    plt.close(fig)

    # Save the data.
    os.makedirs(results_dir, exist_ok=True)
    np.savez(
        results_dir + "sim2_data.npz",
        T_state_gt=T_state_gt,
        T_state_clean=T_state_clean,
        T_state_est=T_state_est,
        T_image_raw=T_image_raw,
        T_image_clean=T_image_clean,
        T_image_est=T_image_est,
        T_rudder_clean=T_rudder_clean,
        T_rudder=T_rudder,
    )


def dynamics(x, y, theta, phi_deg, dt=0.05, v=5, L=5):
    """Dubin's car dynamics model (returns next state)

    Args:
        x: current crosstrack error (meters)
        y: current downtrack position (meters)
        theta: current heading error (degrees)
        phi_deg: steering angle input (degrees)
        -------------------------------
        dt: time step (seconds)
        v: speed (m/s)
        L: distance between front and back wheels (meters)
    """

    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi_deg)

    x_dot = v * np.sin(theta_rad)
    y_dot = v * np.cos(theta_rad)
    theta_dot = (v / L) * np.tan(phi_rad)

    x_prime = x + x_dot * dt
    y_prime = y + y_dot * dt
    theta_prime = theta + np.rad2deg(theta_dot) * dt

    return x_prime, theta_prime, y_prime


def save_results(results_dict):
    T_t = results_dict['T_t']
    T_state_gt = results_dict['T_state_gt']
    T_state_clean = results_dict['T_state_clean']
    T_state_est = results_dict['T_state_est']
    T_rudder_clean = results_dict['T_rudder_clean']
    T_rudder = results_dict['T_rudder']
    T_rudder_filtered = results_dict['T_rudder_filtered']
    T_image_raw = results_dict.get('T_image_raw', None)
    T_image_clean = results_dict['T_image_clean']
    T_image_est = results_dict['T_image_est']
    T_state_bounds = results_dict.get('T_state_bounds', None)
    lows = np.array([bound.lo for bound in T_state_bounds])
    highs = np.array([bound.hi for bound in T_state_bounds])
    lows_ = lows + np.array([settings.CTE_BUFFER, jnp.deg2rad(settings.HE_BUFFER)])
    highs_ = highs - np.array([settings.CTE_BUFFER, jnp.deg2rad(settings.HE_BUFFER)])

    T_state_gt = np.stack(T_state_gt, axis=0)
    T_state_clean = np.stack(T_state_clean, axis=0)
    T_state_est = np.stack(T_state_est, axis=0)
    if T_image_raw is not None:
        T_image_raw = np.stack(T_image_raw, axis=0)
    T_image_clean = np.stack(T_image_clean, axis=0)
    T_image_est = np.stack(T_image_est, axis=0)

    # Set up plot vars
    labels = ["CTE (m)", "DTP (m)", "HE (degrees)"]
    rgb_colors = np.array(pypalettes.load_cmap("Alexandrite").rgb)/255.0
    black = list(rgb_colors[0])
    pink = list(rgb_colors[1])
    dark_teal = list(rgb_colors[2])
    light_teal = list(rgb_colors[3])
    purple = list(rgb_colors[4])
    orange = list(rgb_colors[5])
    light_purple = list(rgb_colors[6])
    teal = list(rgb_colors[7])
    # Plot.
    fig, axes = plt.subplots(2, layout="constrained")
    for ii, ax in enumerate(axes):
        ax.plot(T_t, T_state_gt[:, [0, 2]][:, ii], color=dark_teal, label="True", linewidth=1)
        ax.plot(T_t, T_state_est[:, [0, 2]][:, ii], color=pink, label="Estimated", linewidth=1)
        ax.plot(T_t, T_state_clean[:, [0, 2]][:, ii], color=purple, label="No attack", linewidth=1)
        if settings.FILTER:
            if ii == 1:
                lows[:, ii] = np.rad2deg(lows[:, ii])
                highs[:, ii] = np.rad2deg(highs[:, ii])
                lows_[:, ii] = np.rad2deg(lows_[:, ii])
                highs_[:, ii] = np.rad2deg(highs_[:, ii])
            ax.fill_between(T_t, lows[:, ii], highs[:, ii], color=light_teal, alpha=0.4, label="State bounds" if ii == 0 else None)
            ax.fill_between(T_t, lows_[:, ii], highs_[:, ii], color=teal, alpha=0.4, label="Filtered bounds" if ii == 0 else None)
        ax.set_ylabel(labels[ii], rotation=0, ha="right")
    axes[0].legend()
    fig.savefig(results_dir + "sim2_traj.pdf")
    plt.show()
    plt.close(fig)
    # fig, axes = plt.subplots(3, layout="constrained")
    # for ii, ax in enumerate(axes):
    #     ax.plot(T_t, T_state_gt[:, ii], color="C4", label="True")
    #     ax.plot(T_t, T_state_est[:, ii], color="C1", label="Estimated")
    #     ax.plot(T_t, T_state_clean[:, ii], color="C0", label="No attack")
    #     ax.set_ylabel(labels[ii], rotation=0, ha="right")
    # axes[2].legend()
    # fig.savefig(results_dir + "sim2_traj.pdf")
    # plt.show()
    # plt.close(fig)
    ###############################################3
    # x axis = DTP, y axis = CTE.
    cte_constr = 10.0
    ylim = 11.0
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_state_gt[:, 1], T_state_gt[:, 0], marker="o", lw=0.5, color="C1")
    ax.set_aspect("equal")
    ymin, ymax = ax.get_ylim()
    ymin, ymax = min(ymin, -ylim), max(ymax, ylim)
    ax.set_ylim(ymin, ymax)
    ax.axhspan(cte_constr, ymax, color="C0", alpha=0.2)
    ax.axhspan(-cte_constr, ymin, color="C0", alpha=0.2)
    fig.savefig(results_dir + "sim2_plot2d.pdf")
    plt.show()
    plt.close(fig)
    ###############################################3
    # rudder vs attacked rudder
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_t, T_rudder, color="C1", label="Attacked")
    ax.plot(T_t, T_rudder_clean, color="C4", label="Clean")
    ax.legend()
    # ymin, ymax = ax.get_ylim()
    # ymin, ymax = min(ymin, -ylim), max(ymax, ylim)
    # ax.set_ylim(ymin, ymax)
    # ax.axhspan(cte_constr, ymax, color="C0", alpha=0.2)
    # ax.axhspan(-cte_constr, ymin, color="C0", alpha=0.2)
    fig.savefig(results_dir + "rudder.pdf")
    plt.show()
    plt.close(fig)

    print("Error Metrics:")
    print("CTE MAE:", np.nanmean(np.abs(T_state_gt[:, 0] - T_state_clean[:, 0])))
    # print("CTE RMSE:", np.nansqrt(np.mean((T_state_gt[:, 0] - T_state_est[:, 0])**2)))
    print("CTE MaxAE:", np.nanmax(np.abs(T_state_gt[:, 0] - T_state_clean[:, 0])))
    print("HE MAE:", np.nanmean(np.abs(T_state_gt[:, 2] - T_state_clean[:, 2])))
    # print("HE RMSE:", np.nansqrt(np.mean((T_state_gt[:, 2] - T_state_est[:, 2])**2)))
    print("HE MaxAE:", np.nanmax(np.abs(T_state_gt[:, 2] - T_state_clean[:, 2])))


    # Save the data.
    
    with open(results_dir + "sim2_results.pkl", "wb") as f:
        pickle.dump(results_dict, f)
    
    settings_dict = {k: v for k, v in settings.__dict__.items() if not k.startswith("__") and not callable(v)}
    with open(results_dir + "settings.json", "w") as f:
        json.dump(settings_dict, f, indent=2, default=str)


def simulate_controller_dubins(
    client, startCTE, startHE, startDTP, endDTP, get_state, get_control, dt, ctrlEvery, simSpeed=1.0, results_dict={}
):
    """Simulates a controller, overriding the built-in XPlane-11 dynamics to model the aircraft
    as a Dubin's car

    Args:
        client: XPlane Client
        startCTE: Starting crosstrack error (meters)
        startHE: Starting heading error (degrees)
        startDTP: Starting downtrack position (meters)
        endDTP: Ending downtrack position (meters)
        getState: Function to estimate the current crosstrack and heading errors.
                  Takes in an XPlane client and returns the crosstrack and
                  heading error estimates
        getControl: Function to perform control based on the state
                    Takes in an XPlane client, the current crosstrack error estimate,
                    and the current heading error estimate and returns a control effort
        dt: time step (seconds)
        crtlEvery: Frequency to get new control input
                   (e.g. if dt=0.5, a value of 20 for ctrlEvery will perform control
                   at a 1 Hz rate)
        -------------------
        simSpeed: increase beyond 1 to speed up simulation
    """
    # # Reset to the desired starting position
    # client.sendDREF("sim/time/sim_speed", simSpeed)
    # xpc3_helper.reset(client, cteInit=startCTE, heInit=startHE, dtpInit=startDTP)
    # xpc3_helper.sendBrake(client, 0)

    # time.sleep(5)  # 5 seconds to get terminal window out of the way

    # cte = startCTE
    # he = startHE
    # dtp = startDTP
    # startTime = client.getDREF("sim/time/zulu_time_sec")[0]
    # endTime = startTime

    # while dtp < endDTP:
    #     image_raw = get_xplane_image()
    #     cte_pred, he_pred, img = get_state(image_raw, attack=settings.ATTACK)
    #     phiDeg = get_control(client, cte_pred, he_pred)

    #     for i in range(ctrlEvery):
    #         cte, he, dtp = dynamics(cte, dtp, he, phiDeg, dt)
    #         xpc3_helper.setHomeState(client, cte, dtp, he)
    #         time.sleep(0.03)

    # client.pauseSim(True)

    client.sendDREF("sim/time/sim_speed", simSpeed)
    xpc3_helper.reset(client, cteInit=startCTE, heInit=startHE, dtpInit=startDTP)
    xpc3_helper.sendBrake(client, 0)
    key = jax.random.PRNGKey(0)
    settings.NETWORK()  # Initialize the network once before starting the sim loop

    if settings.FILTER:
        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
            hj.sets.Box(
                np.array([-18., -np.pi/3]),
                np.array([18., np.pi/3])),
            (100, 100),
        )
        values = -jnp.abs(grid.states[..., 0]) + 10.0
        filter_dyn = dynamic_models.TaxiNetDynamics(
            dt=settings.DT,
            max_rudder=jnp.tan(jnp.deg2rad(settings.MAX_RUDDER))
        )
        # Set up the uncertainty-aware filter
        if settings.SMOOTHING:
            hjnnv_filter = hjnnvUncertaintyAwareFilter(
                filter_dyn,
                pred_model=settings.GET_STATE_SMOOTHED,
                grid=grid,
                initial_values=values,
                num_controls=30,
                num_disturbances=15,
            )
        else:
            hjnnv_filter = hjnnvUncertaintyAwareFilter(
                filter_dyn,
                pred_model=settings.GET_STATE,
                grid=grid,
                initial_values=values,
                num_controls=30,
                num_disturbances=15,
            )

        # Use random calculations to jit compile things before the main loop
        v_star, u_star, worst_val, val_filter, _, _, _, _ = hjnnv_filter.ua_filter_best_u(
                jnp.array(np.tan(np.deg2rad(0))),
                hj.sets.Box(
                    jnp.array([-0.1, -0.1]),
                    jnp.array([0.1, 0.1])
                ),
                num_states=10,
        )
        if settings.STATE_ESTIMATOR == 'tiny_taxinet':
            if settings.SMOOTHING:
                dummy_input = jnp.zeros((2 + 1 + 128,))
            else:
                dummy_input = jnp.zeros((128,))

        elif settings.STATE_ESTIMATOR == 'dnn':
            dummy_input = jnp.zeros((1, 3, 224, 224))
        elif settings.STATE_ESTIMATOR in ['cnn64', 'cnn']:
            if settings.SMOOTHING:
                dummy_input = jnp.zeros((1+1, 64, 64))
            else:
                dummy_input = jnp.zeros((1, 64, 64))

        # p = settings.NETWORK()   # or settings.GET_STATE_SMOOTHED
        # print("pred_model callable?", callable(p))
        # # If pred_model is a closure, try:
        # try:
        #     from inspect import getsource
        #     print(getsource(p))
        # except Exception:
        #     pass
        # # Check dummy input is concrete
        # print("dummy_input type:", type(dummy_input), "dtype:", getattr(dummy_input, "dtype", None))

        # with jax.checking_leaks():
        state_bounds = hjnnv_filter.nnv_state_bounds(
            dummy_input,
            0.03
        )

    

        # @jax.jit
        # def val_filter_extract(val_filter):
        #     return float(val_filter)
        # val_filter = val_filter_extract(val_filter)

    def host_scalar(x):
        # Works for shape-() JAX/NumPy arrays and Python floats
        return float(np.asarray(x))
    
    # client.pauseSim(False)

    # client.sendDREF("sim/time/sim_speed", simSpeed)
    # xpc3_helper.reset(client, cteInit=startCTE, heInit=startHE, dtpInit=startDTP)
    # xpc3_helper.sendBrake(client, 0)

    dtp = startDTP
    startTime = client.getDREF("sim/time/zulu_time_sec")[0]
    now = startTime
    endTime = startTime
    run_end_time = now + 1000.0

    T_t = results_dict.get('T_t', [])
    T_state_gt = results_dict.get('T_state_gt', [])
    T_state_clean = results_dict.get('T_state_clean', [])
    T_state_est = results_dict.get('T_state_est', [])
    T_rudder_clean = results_dict.get('T_rudder_clean', [])
    T_rudder = results_dict.get('T_rudder', [])
    T_rudder_filtered = results_dict.get('T_rudder_filtered', [])
    T_rudder_unfiltered = results_dict.get('T_rudder_unfiltered', [])
    T_image_raw = results_dict.get('T_image_raw', [])
    T_image_clean = results_dict.get('T_image_clean', [])
    T_image_est = results_dict.get('T_image_est', [])
    T_state_bounds = results_dict.get('T_state_bounds', [])
    T_filter_time = results_dict.get('T_filter_time', [])
    # ipdb.set_trace()

    cte_gt, dtp_gt, he_gt = xpc3_helper.getHomeState(client)
    cte_dym, dtp_dym, he_dym = cte_gt, dtp_gt, he_gt

    dt = settings.DT
    filtering = settings.FILTER

    image_raw = get_xplane_image()
    cte_clean, he_clean, img_clean, adv_image = get_state(image_raw, key=key)
    adv_img = None

    cte_pred, he_pred = cte_clean, he_clean
    ctrl_h = 0.0
    init_sim_len = len(T_state_gt)


    def rudder_target(pos, target_rudder=-1.0):
        cte, he = pos
        rudder = jnp.clip(-0.74 * cte - 0.44 * he, -7.0, 7.0)
        return (rudder - target_rudder)**2

    while dtp_dym < endDTP and now < run_end_time and np.abs(cte_gt) < 10.5:
        key, subkey = jax.random.split(key)
        simLoopTimer = time.time()
        x_hat_prev = jnp.array([cte_pred, he_pred])
        x_hat_prev_clean = jnp.array([cte_clean, he_clean])
        u_hat_prev = jnp.array([jnp.tan(jnp.deg2rad(ctrl_h))])
        # Get ground truth state
        cte_gt, dtp_gt, he_gt = xpc3_helper.getHomeState(client)
        state_gt = np.array([cte_gt, dtp_gt, he_gt])
        T_state_gt.append(state_gt)
        print(len(T_state_gt))

        image_raw = get_xplane_image()
        T_image_raw.append(image_raw)

        # Get the estimated state and control without attack
        # Need to think about this line more - should not be using x_prev_clean, but what is best thing to plot?
        # Plotting should be x_hat_prev, but error buffer should be calculated by x_hat_prev_clean
        cte_clean, he_clean, img_clean, _ = get_state(image_raw, x_prev=x_hat_prev, u_prev=u_hat_prev, key=subkey)
        phiDeg_clean = get_control(client, cte_clean, he_clean)
        state_bounds = hj.sets.Box(lo=jnp.empty((2,)), hi=jnp.empty((2,)))

        target_rudder = settings.TARGET_FUNCTION(cte_gt, he_gt, settings.MAX_RUDDER)
        print("Target for attack:", target_rudder)
        
        cte_pred, he_pred, img, adv_img = get_state(image_raw, x_prev=x_hat_prev, u_prev=u_hat_prev, target=target_rudder, attack=rudder_target, adv_ptb_prev=adv_img, key=subkey)
        logger.info("----------------------------------------------------------")
        logger.info("CTE: {:.2f} ({:.2f}), HE: {:.2f} ({:.2f})".format(cte_pred, cte_gt, he_pred, he_gt))
        phiDeg = get_control(client, cte_pred, he_pred)
        ctrl = phiDeg
        # Filter debugging with bad controller
        # phiDeg = phiDeg*0
        # ipdb.set_trace()

        # Use the uncertainty-aware filter to get the control input
        time_start = time.time()
        if filtering:
            time_start1 = time.time()
            if settings.STATE_ESTIMATOR == 'tiny_taxinet':
                print(img.shape)
                # img = img.reshape(-1, 1)
            elif settings.STATE_ESTIMATOR == 'dnn':
                img = img.reshape(-1, 3, 224, 224)
            # print("Time taken for reshaping: {:.5f}".format(time.time() - time_start1))
            time_start2 = time.time()
            if settings.SMOOTHING and settings.STATE_ESTIMATOR == 'tiny_taxinet':
                # obs = jnp.concatenate([x_hat_prev, u_hat_prev, img])
                obs = settings.PACKAGE_INPUT(x_hat_prev, u_hat_prev, img)
                eps = jnp.concatenate([jnp.ones((3,))*1e-6, jnp.ones((128,))*settings.ATTACK_STRENGTH])
            elif settings.SMOOTHING and settings.STATE_ESTIMATOR in ['cnn64', 'cnn']:
                # obs = jnp.concatenate([x_hat_prev, jnp.expand_dims(img, axis=0)], axis=0)
                obs = settings.PACKAGE_INPUT(x_hat_prev, u_hat_prev, img)
                eps = jnp.concatenate([jnp.ones(img.shape)*0.0, jnp.ones(img.shape)*settings.ATTACK_STRENGTH], axis=0)

            else:
                obs = img
                eps = settings.ATTACK_STRENGTH

            # Right before calling nnv_state_bounds(...)
            print("nnv input shape:", obs.shape)
            print("nnv eps shape:", getattr(eps, "shape", None), "eps value:", eps if getattr(eps, "shape", None) is None else None)
            print("obs (first 10):", obs.flatten()[:10])
            if hasattr(eps, "shape"):
                print("eps (first 10):", eps.flatten()[:10])

            # If using smoothing, ensure the predicted-smoothing function accepts that exact shape:
            try:
                # run the point-prediction for the same obs you pass
                pred_point = settings.GET_STATE_SMOOTHED(obs)   # or settings.GET_STATE_SMOOTHED(obs) if outside class
                print("pred_point:", pred_point)
            except Exception as e:
                print("pred_model raised:", e)
            
            state_bounds = hjnnv_filter.nnv_state_bounds(
                obs,
                eps
            )

            lo = jnp.array([state_bounds.lo[0], jnp.deg2rad(state_bounds.lo[1])])
            hi = jnp.array([state_bounds.hi[0], jnp.deg2rad(state_bounds.hi[1])])
            cte_buffer = settings.CTE_BUFFER
            heading_buffer = jnp.deg2rad(settings.HE_BUFFER)
            # else:
            #     pos_buffer = 0
            #     vel_buffer = 0
            state_bounds = hj.sets.Box(
                lo=lo - jnp.array([cte_buffer, heading_buffer]),
                hi=hi + jnp.array([cte_buffer, heading_buffer])
            )
            # state_bounds = hjnnv_filter.state_bounds_from_gt(
            #     jnp.array([cte_pred, np.deg2rad(he_pred)]),
            #     jnp.array([cte_gt, np.deg2rad(he_gt)])
            # )
            # ipdb.set_trace()
            time_nnv = time.time() - time_start2
            # print("Time taken for NNV: {:.5f}".format(time_nnv))
            time_start3 = time.time()
            v_star, tan_phi_star_rad, worst_val, val_filter, _, _, _, _, = hjnnv_filter.ua_filter_best_u(
                jnp.array(np.tan(np.deg2rad(phiDeg))),
                state_bounds,
                num_states=15,
            )
            
            time_filter = time.time() - time_start3
            # print("Time taken for filtering: {:.5f}".format(time_filter))
            
            time_convert = time.time()
            phiDeg_star = jnp.rad2deg(jnp.arctan(tan_phi_star_rad))
            # print("Time taken for rad2deg: {:.5f}".format(time.time() - time_convert))

            time_check = time.time()
            # ipdb.set_trace()
            # # val_filter = float(val_filter)
            # if val_filter < 0 and filtering:
            #     ctrl = phiDeg_star
            ctrl = jnp.where((val_filter < 0) & filtering, phiDeg_star, ctrl)
            # print("Time taken for val check: {:.5f}".format(time.time() - time_check))

            

            # x_lo  = host_scalar(state_bounds.lo[0])
            # x_hi  = host_scalar(state_bounds.hi[0])
            # th_lo = host_scalar(state_bounds.lo[1])
            # th_hi = host_scalar(state_bounds.hi[1])

            # val_filter_h = host_scalar(val_filter)
            # v_star_h     = host_scalar(v_star)
            # phiDeg_h     = host_scalar(phiDeg)       # if this is JAX
            
            # phiDeg_star_h= host_scalar(phiDeg_star)

            # ipdb.set_trace()

            time_logs = time.time()
            # logger.info("Filter Time: {:.5f}, NNV Time: {:.5f}".format(time_filter, time_nnv))
            logger.info("x bounds: {:.2f} <-> {:.2f}, theta bounds: {:.2f} <-> {:.2f}".format(
                state_bounds.lo[0], state_bounds.hi[0], jnp.rad2deg(state_bounds.lo[1]), jnp.rad2deg(state_bounds.hi[1])
            ))
            # logger.info("Nominal Value: {:.2f}, Optimal value: {:.2f}".format(val_filter, v_star))
            # logger.info("Control: {:.2f}, Filtered Control: {:.2f}".format(phiDeg, ctrl))
            # logger.info("Filter Time: %.5f, NNV Time: %.5f", time_filter, time_nnv)
            # logger.info("x bounds: %.2f <-> %.2f, theta bounds: %.2f <-> %.2f",
            #             x_lo, x_hi, th_lo, th_hi)
            # logger.info("Nominal Value: %.2f, Optimal value: %.2f",
            #             val_filter_h, v_star_h)
            # logger.info("Control: %.2f, Filtered Control: %.2f",
            #             phiDeg_h, ctrl_h)
            T_rudder_filtered.append(phiDeg_star)
            T_rudder_unfiltered.append(phiDeg)
            # print("Time taken for logging: {:.5f}".format(time.time() - time_logs))
        time_filter_total = time.time() - time_start
        logger.info("Filter Total Time: {:.5f}".format(time_filter_total))

        T_rudder_clean.append(phiDeg_clean)
        T_rudder.append(phiDeg)

        T_image_clean.append(img_clean)
        T_image_est.append(img)

        state_est = np.array([cte_pred, dtp_gt, he_pred])
        T_state_est.append(state_est)
        state_clean = np.array([cte_clean, dtp_gt, he_clean])
        T_state_clean.append(state_clean)

        T_state_bounds.append(state_bounds)

        T_filter_time.append(time_filter_total)

        # # Wait for next timestep. 1 Hz control rate?
        # while endTime - startTime < dt:
        #     endTime = client.getDREF("sim/time/zulu_time_sec")[0]
        #     time.sleep(0.001)

        # # Set things for next round
        # startTime = client.getDREF("sim/time/zulu_time_sec")[0]
        # endTime = startTime
        # _, dtp, _ = xpc3_helper.getHomeState(client)
        # time.sleep(0.001)
        # now = startTime
        now = client.getDREF("sim/time/zulu_time_sec")[0]
        ctrl_h = host_scalar(ctrl)
        for i in range(ctrlEvery):
            cte_dym, he_dym, dtp_dym = dynamics(cte_dym, dtp_dym, he_dym, ctrl_h, dt)
            xpc3_helper.setHomeState(client, cte_dym, dtp_dym, he_dym)
            time.sleep(0.03)
        logger.info("Simulation loop time: {:.5f} seconds".format(time.time() - simLoopTimer))

    client.pauseSim(True)

    print("Filter time average: {:.5f} s".format(np.mean(T_filter_time[1:])))
    print("Filter time stdev: {:.5f} s".format(np.std(T_filter_time[1:])))
    print("Simulation ended. Saving results...")

    if results_dict == {}:
        T_t = np.arange(len(T_state_gt)) * dt
        # T_t = np.concatenate([T_t, np.array([np.nan])])
    else:
        T_t_ = np.arange(len(T_state_gt) - init_sim_len) * dt
        # T_t = np.concatenate([results_dict['T_t'], T_t_, np.array([np.nan])])

    # T_state_gt.append([np.nan]*state_gt.shape[0])
    # T_state_clean.append([np.nan]*state_gt.shape[0])
    # T_state_est.append([np.nan]*state_gt.shape[0])
    # T_rudder_clean.append(np.nan)
    # T_rudder.append(np.nan)
    # T_rudder_filtered.append(np.nan)

    results_dict = {
        "results_dir": results_dir,
        "T_t": T_t,
        "T_state_gt": T_state_gt,
        "T_state_clean": T_state_clean,
        "T_state_est": T_state_est,
        # "T_image_raw": T_image_raw,
        "T_image_clean": T_image_clean,
        "T_image_est": T_image_est,
        "T_rudder_clean": T_rudder_clean,
        "T_rudder": T_rudder,
        "T_rudder_filtered": T_rudder_filtered,
        "T_rudder_unfiltered": T_rudder_unfiltered,
        "T_state_bounds": T_state_bounds,
        "value_function": hjnnv_filter.target_values if filtering else None,
    }

    os.makedirs(results_dir, exist_ok=True)
    save_results(results_dict)

    return results_dict


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
