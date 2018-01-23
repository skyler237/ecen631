#!/usr/bin/env python
import numpy as np
import math
import transforms3d as tf3d
import pyqtgraph as pg
from pyqtgraph import ViewBox

from Holodeck.Sensors import Sensors

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

class Plotter:
    """
    Class for plotting methods.
    """
    def __init__(self):
        # get parameters from server
        self.t_win = 5.0
        self.time0 = 0
        self.init_time = True
        self.curve_length = 300
        self.time_window = 15.0

        # initialize Qt gui application and window
        self.app = pg.QtGui.QApplication([])
        self.w = pg.GraphicsWindow(title='Holodeck UAV States')
        self.w.resize(1000,800)

        # initialize plots in one window
        # Position plots
        self.p_x = self.w.addPlot()
        self.p_x.setLabel('left', 'x')
        self.p_y = self.w.addPlot()
        self.p_y.setLabel('left', 'y')
        self.p_z = self.w.addPlot()
        self.p_z.addLegend(size=(1,1), offset=(1,1))
        self.p_z.setLabel('left', 'z')
        self.w.nextRow()

        # Velocity plots
        self.p_xdot = self.w.addPlot()
        self.p_xdot.setLabel('left', 'xdot')
        self.p_ydot = self.w.addPlot()
        self.p_ydot.setLabel('left', 'ydot')
        self.p_zdot = self.w.addPlot()
        self.p_zdot.setLabel('left', 'zdot')
        self.w.nextRow()

        # Attitude plots
        self.p_phi = self.w.addPlot()
        self.p_phi.setLabel('left', 'phi')
        self.p_theta = self.w.addPlot()
        self.p_theta.setLabel('left', 'theta')
        self.p_psi = self.w.addPlot()
        self.p_psi.setLabel('left', 'psi')
        self.w.nextRow()

        # Angular velocity plots
        self.p_p = self.w.addPlot()
        self.p_p.setLabel('left', 'p')
        self.p_q = self.w.addPlot()
        self.p_q.setLabel('left', 'q')
        self.p_r = self.w.addPlot()
        self.p_r.setLabel('left', 'r')
        self.w.nextRow()

        # Acceleration plots
        self.p_ax = self.w.addPlot()
        self.p_ax.setLabel('left', 'ax')
        self.p_ay = self.w.addPlot()
        self.p_ay.setLabel('left', 'ay')
        self.p_az = self.w.addPlot()
        self.p_az.setLabel('left', 'az')
        self.w.nextRow()


        # Actuals curves
        self.c_x = self.p_x.plot()
        self.c_y = self.p_y.plot()
        self.c_z = self.p_z.plot(name='Actual')
        self.c_xdot = self.p_xdot.plot()
        self.c_ydot = self.p_ydot.plot()
        self.c_zdot = self.p_zdot.plot()
        self.c_phi = self.p_phi.plot()
        self.c_theta = self.p_theta.plot()
        self.c_psi = self.p_psi.plot()
        self.c_p = self.p_p.plot()
        self.c_q = self.p_q.plot()
        self.c_r = self.p_r.plot()
        self.c_ax = self.p_ax.plot()
        self.c_ay = self.p_ay.plot()
        self.c_az = self.p_az.plot()

        # Command curves
        self.c_z_c = self.p_z.plot(name='Commanded')
        self.c_phi_c = self.p_phi.plot()
        self.c_theta_c = self.p_theta.plot()
        self.c_r_c = self.p_r.plot()

        # initialize state variables
        self.time = 0.0
        self.prev_time = -1

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.xdot = 0.0
        self.ydot = 0.0
        self.zdot = 0.0
        self.phi = 0.0
        self.theta = 0.0
        self.psi = 0.0
        self.p = 0.0
        self.q = 0.0
        self.r = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0

        self.z_c = 0.0
        self.phi_c = 0.0
        self.theta_c = 0.0
        self.r_c = 0.0

        self.states = []
        self.commands = []

        # plot list
        self.p_list = [self.p_x,self.p_y,self.p_z,self.p_xdot,self.p_ydot,self.p_zdot,self.p_phi,self.p_theta,self.p_psi,self.p_p,self.p_q,self.p_r,self.p_ax,self.p_ay,self.p_az]
        for plot in self.p_list:
            state = plot.getViewBox().getState()
            state["autoVisibleOnly"] = [False, True]
            plot.getViewBox().setState(state)

        # curve lists
        self.c_list = [self.c_x,self.c_y,self.c_z,self.c_xdot,self.c_ydot,self.c_zdot,self.c_phi,self.c_theta,self.c_psi,self.c_p,self.c_q,self.c_r,self.c_ax,self.c_ay,self.c_az]
        self.c_list_c = [self.c_z_c,self.c_phi_c,self.c_theta_c,self.c_r_c]

    # Update the plots with the current data
    def update_plots(self):
        if self.prev_time != self.time:
            # pack stored data into lists
            self.states.append([self.time, self.x, self.y, self.z, self.xdot, self.ydot, self.zdot, self.phi, self.theta, self.psi, self.p, self.q, self.r, self.ax, self.ay, self.az])
            self.commands.append([self.time, self.z_c, self.phi_c, self.theta_c, self.r_c])

            # stack the data lists
            states_array = np.vstack(self.states)
            time_array = states_array[:,0]

            commands_array = np.vstack(self.commands)

            # set the truth states
            for i in range(0,len(self.c_list)):
    	        self.c_list[i].setData(time_array, states_array[:,i+1], pen=(255,0,0))

            # set the estimated states
            for i in range(0,len(self.c_list_c)):
                self.c_list_c[i].setData(time_array, commands_array[:,i+1], pen=(0,255,0))

            x_min = max(self.time - self.time_window, 0)
            x_max = self.time
            for i in range(0,len(self.p_list)):
                self.p_list[i].setXRange(x_min, x_max)
                self.p_list[i].enableAutoRange(axis=ViewBox.YAxis)

        self.prev_time = self.time
        # update the plotted data
        self.app.processEvents()


    def update_states(self, state, time):
        [self.x, self.y, self.z] = state[Sensors.LOCATION_SENSOR]
        [self.xdot, self.ydot, self.zdot] = state[Sensors.VELOCITY_SENSOR]
        [self.p, self.q, self.r, self.ax, self.ay, self.az] = state[Sensors.IMU_SENSOR]
        rot_mat = state[Sensors.ORIENTATION_SENSOR]
        [self.phi, self.theta, self.psi] = tf3d.euler.mat2euler(rot_mat, 'rxyz')
        self.time = time

    def update_commands(self, command):
        self.phi_c = -command[0] # For some reason, the command seems backward on this...
        self.theta_c = command[1]
        self.r_c = command[2]
        self.z_c = command[3]
