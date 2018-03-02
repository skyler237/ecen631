import numpy as np
import math
import transforms3d

from Holodeck.Sensors import Sensors
from Plotter import Plotter

class PlotterWrapper:
    ''' Plotter wrapper  '''
    def __init__(self, plotting_freq):
        self.plotter = Plotter(plotting_freq)

        # Define plot names
        plots = self._define_plots()
        legends = self._define_legends()

        # Add plots to the window
        for p in plots:
            self.plotter.add_plot(p, include_legend=(p[0] in legends))

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

    def _define_legends(self):
        legends = []
        return legends

    def _define_input_vectors(self):
        pass

    def update_plot_data(self, uav_sim):
        pass

class CommandsPlotter(PlotterWrapper):
    ''' Plotter wrapper for viewing commanded values '''
    # NOTE: Inherits constructor

    def _define_plots(self):
        # Define plot names
        plots = ['x',                   'y',                    ['z', 'z_c'],
                 ['xdot', 'xdot_c'],    ['ydot', 'ydot_c'],     'zdot',
                 ['phi', 'phi_c'],      ['theta', 'theta_c'],   ['psi', 'psi_c'],
                 'p',                   'q',                    ['r', 'r_c'],
                 'ax',                  'ay',                   'az'
                 ]
        return plots

    def _define_legends(self):
        return ['z']

    def _define_input_vectors(self):
        self.plotter.define_input_vector("position", ['x', 'y', 'z'])
        self.plotter.define_input_vector("velocity", ['xdot', 'ydot', 'zdot'])
        self.plotter.define_input_vector("orientation", ['phi', 'theta', 'psi'])
        self.plotter.define_input_vector("imu", ['ax', 'ay', 'az', 'p', 'q', 'r'])
        self.plotter.define_input_vector("command", ['phi_c', 'theta_c', 'r_c', 'z_c'])
        self.plotter.define_input_vector("vel_command", ['xdot_c', 'ydot_c', 'psi_c'])

    def update_sim_data(self, uav_sim):
        t = uav_sim.get_sim_time()
        self.plotter.add_vector_measurement("position",     uav_sim.get_position(), t)
        self.plotter.add_vector_measurement("velocity",     uav_sim.get_body_velocity(), t)
        self.plotter.add_vector_measurement("orientation",  uav_sim.get_euler(), t)
        self.plotter.add_vector_measurement("imu",          uav_sim.get_imu(), t)
        self.plotter.add_vector_measurement("command",      uav_sim.get_sim_command(), t)
        self.plotter.add_vector_measurement("vel_command",  uav_sim.get_vel_command(), t)
        self.plotter.update_plots()
