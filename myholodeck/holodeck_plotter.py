import numpy as np
import math
import transforms3d

from Holodeck.Sensors import Sensors
from Plotter import Plotter

class HolodeckPlotter:
    ''' Plotter wrapper for holodeck '''
    def __init__(self, plotting_freq):
        self.plotter = Plotter(plotting_freq)

        # Define plot names
        plots = self._define_plots()

        # Add plots to the window
        for p in plots:
            self.plotter.add_plot(p)

        # Define state vectors for simpler input
        self._define_input_vectors()

    def _define_plots(self):
        plots = ['x',       'y',        'z',
                 'xdot',    'ydot',     'zdot',
                 'phi',     'theta',    'psi',
                 'p',       'q',        'r',
                 'ax',      'ay',       'az'
                 ]
        return plots

    def _define_input_vectors(self):
        self.plotter.define_input_vector("position", ['x', 'y', 'z'])
        self.plotter.define_input_vector("velocity", ['xdot', 'ydot', 'zdot'])
        self.plotter.define_input_vector("orientation", ['phi', 'theta', 'psi'])
        self.plotter.define_input_vector("imu", ['ax', 'ay', 'az', 'p', 'q', 'r'])
        pass

    def update_sim_data(self, uav_sim):
        self.t = uav_sim.get_sim_time()
        self.plotter.add_vector_measurement("position",     uav_sim.get_position(), self.t)
        self.plotter.add_vector_measurement("velocity",     uav_sim.get_body_velocity(), self.t)
        self.plotter.add_vector_measurement("orientation",  uav_sim.get_euler(), self.t)
        self.plotter.add_vector_measurement("imu",          uav_sim.get_imu(), self.t)

class CommandsPlotter(HolodeckPlotter):
    ''' Plotter wrapper for viewing commanded values '''
    # NOTE: Inherits constructor

    def _define_plots(self):
        # Define plot names
        plots = ['x',               'y',                'z z_c -l',
                 'xdot xdot_c -l',  'ydot ydot_c -l',   'zdot',
                 'phi phi_c -l',    'theta theta_c -l', 'psi psi_c -l',
                 'p',               'q',                'r r_c',
                 'ax',              'ay',               'az'
                 ]
        return plots

    def _define_input_vectors(self):
        super()._define_input_vectors()
        self.plotter.define_input_vector("command", ['phi_c', 'theta_c', 'r_c', 'z_c'])
        self.plotter.define_input_vector("vel_command", ['xdot_c', 'ydot_c', 'psi_c'])

    def update_sim_data(self, uav_sim):
        super().update_sim_data(uav_sim)
        self.plotter.add_vector_measurement("command",      uav_sim.get_sim_command(), self.t)
        self.plotter.add_vector_measurement("vel_command",  uav_sim.get_vel_command(), self.t)
        self.plotter.update_plots()

class OdometryPlotter(HolodeckPlotter):
    ''' Plotter wrapper for viewing odometry estimated values '''
    # NOTE: Inherits constructor

    def _define_plots(self):
        # Define plot names
        plots = ['x x_e -l',    'y y_e',            'z z_e',
                 'phi phi_e',   'theta theta_e',    'psi psi_e',
                 '_xdot',        '_ydot',             '_zdot',
                 '_p',           '_q',                '_r',
                 '_ax',          '_ay',               '_az',
                 'y x y_e x_e -2d'
                 ]
        return plots

    def _define_input_vectors(self):
        super()._define_input_vectors()
        self.plotter.define_input_vector("position_estimate", ['x_e', 'y_e', 'z_e'])
        self.plotter.define_input_vector("euler_estimate", ['phi_e', 'theta_e', 'psi_e'])

    def update_sim_data(self, uav_sim, position_estimate, euler_estimate):
        super().update_sim_data(uav_sim)
        self.plotter.add_vector_measurement("position_estimate", position_estimate, self.t)
        self.plotter.add_vector_measurement("euler_estimate",  euler_estimate, self.t)
        self.plotter.update_plots()
