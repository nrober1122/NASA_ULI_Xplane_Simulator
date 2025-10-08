import xpc3
import xpc3_helper

from simulators import Simulator
from simulators.NASA_ULI_Xplane_Simulator.src.simulation.run_sim2 import simulate_controller_dubins, get_state
from simulators.NASA_ULI_Xplane_Simulator.src.simulation import settings


class TaxinetSimulator(Simulator):
    def __init__(self, dt=0.1):
        self.experiments = {
            1: {
                "run": self.run_expt_1,
                "plots": {
                    "trajectory": self.plot_expt_1_trajectory,
                    "velocity": self.plot_expt_1_velocity,
                },
            },
        }
        super().__init__(dt=dt)

    # def run(self, num_steps, filtering, bounding_method, experiment_number):
    #     with xpc3_helper.XPlaneConnect() as client:
    #         simulate_controller_dubins(
    #                 client,
    #                 settings.START_CTE,
    #                 settings.START_HE,
    #                 settings.START_DTP,
    #                 settings.END_DTP,
    #                 get_state,
    #                 settings.GET_CONTROL,
    #                 settings.DT,
    #                 settings.CTRL_EVERY,
    #             )

    def plot(self):
        # Implement the plotting logic here
        pass

    def run_expt_1(self):
        with xpc3_helper.XPlaneConnect() as client:
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