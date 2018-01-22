#!/usr/bin/env python
import numpy as np
import math
import pyqtgraph as pg
from pyqtgraph import ViewBox

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

        # Velocity plots

        # Attitude plots

        # Angular velocity plots

        # Acceleration plots


        self.p_bx = self.w.addPlot()
        self.p_by = self.w.addPlot()
        self.p_bz = self.w.addPlot()
        self.w.nextRow()
        self.p_phi = self.w.addPlot()
        self.p_phi.addLegend(size=(1,1), offset=(1,1))
        self.p_theta = self.w.addPlot()
        self.p_psi = self.w.addPlot()
        self.w.nextRow()
        self.p_ax = self.w.addPlot()
        self.p_ay = self.w.addPlot()
        self.p_az = self.w.addPlot()
        self.w.nextRow()
        self.p_p = self.w.addPlot()
        self.p_q = self.w.addPlot()
        self.p_r = self.w.addPlot()
        self.w.nextRow()
        self.p_delta_a = self.w.addPlot()
        self.p_delta_e = self.w.addPlot()
        self.p_delta_r = self.w.addPlot()
        self.w.nextRow()
        self.p_delta_F = self.w.addPlot()
        self.p_Va = self.w.addPlot()
        self.p_Vg = self.w.addPlot()
        self.w.nextRow()
        self.p_alpha = self.w.addPlot()



        # label the plots
        self.p_bx.setLabel('left', 'bx')
        self.p_by.setLabel('left', 'by')
        self.p_bz.setLabel('left', 'bz')
        self.p_phi.setLabel('left', 'phi (deg)')
        self.p_theta.setLabel('left', 'theta (deg)')
        self.p_psi.setLabel('left', 'psi (deg)')
        self.p_ax.setLabel('left', 'ax')
        self.p_ay.setLabel('left', 'ay')
        self.p_az.setLabel('left', 'az')
        self.p_p.setLabel('left', 'p')
        self.p_q.setLabel('left', 'q')
        self.p_r.setLabel('left', 'r')
        self.p_delta_a.setLabel('left', 'delta_a')
        self.p_delta_e.setLabel('left', 'delta_e')
        self.p_delta_r.setLabel('left', 'delta_r')
        self.p_delta_F.setLabel('left', 'delta_F')
        self.p_Va.setLabel('left', 'Va')
        self.p_Vg.setLabel('left', 'Vg')
        self.p_alpha.setLabel('left', 'alpha (deg)')

        # create curves to update later
        self.c_bx_t = self.p_bx.plot()
        self.c_by_t = self.p_by.plot()
        self.c_bz_t = self.p_bz.plot()
        self.c_ax_t = self.p_phi.plot(name='Truth')
        self.c_ay_t = self.p_theta.plot()
        self.c_az_t = self.p_psi.plot()
        self.c_phi_t = self.p_ax.plot()
        self.c_theta_t = self.p_ay.plot()
        self.c_psi_t = self.p_az.plot()
        self.c_p_t = self.p_p.plot()
        self.c_q_t = self.p_q.plot()
        self.c_r_t = self.p_r.plot()
        self.c_delta_a_t = self.p_delta_a.plot()
        self.c_delta_e_t = self.p_delta_e.plot()
        self.c_delta_r_t = self.p_delta_r.plot()
        self.c_delta_F_t = self.p_delta_F.plot()
        self.c_Va_t = self.p_Va.plot()
        self.c_Vg_t = self.p_Vg.plot()
        self.c_alpha_t = self.p_alpha.plot()

        self.c_bx_e = self.p_bx.plot()
        self.c_by_e = self.p_by.plot()
        self.c_bz_e = self.p_bz.plot()
        self.c_ax_e = self.p_phi.plot(name='rosplane estimate')
        self.c_ay_e = self.p_theta.plot()
        self.c_az_e = self.p_psi.plot()
        self.c_phi_e = self.p_ax.plot()
        self.c_theta_e = self.p_ay.plot()
        self.c_psi_e = self.p_az.plot()
        self.c_p_e = self.p_p.plot()
        self.c_q_e = self.p_q.plot()
        self.c_r_e = self.p_r.plot()
        self.c_delta_a_e = self.p_delta_a.plot()
        self.c_delta_e_e = self.p_delta_e.plot()
        self.c_delta_r_e = self.p_delta_r.plot()
        self.c_delta_F_e = self.p_delta_F.plot()
        self.c_Va_e = self.p_Va.plot()
        self.c_Vg_e = self.p_Vg.plot()
        self.c_alpha_e = self.p_alpha.plot()
        # EKF attitude estimates
        self.c_phi_e2 = self.p_phi.plot(name='MEKF estimate')
        self.c_theta_e2 = self.p_theta.plot()
        self.c_psi_e2 = self.p_psi.plot()

        # initialize state variables
        self.time_t = 0.0
        self.prev_time = -999.0
        # self.bx_t = 1.0 / 180.0 * math.pi
        # self.by_t = 1.0 / 180.0 * math.pi
        # self.bz_t = 1.0 / 180.0 * math.pi
        bias = 0.0 # (deg.)
        self.bx_t = bias
        self.by_t = -bias
        self.bz_t = -bias
        self.ax_t = 0
        self.ay_t = 0
        self.az_t = 0
        self.phi_t = 0
        self.theta_t = 0
        self.psi_t = 0
        self.p_t = 0
        self.q_t = 0
        self.r_t = 0
        self.p_imu = 0
        self.q_imu = 0
        self.r_imu = 0
        self.delta_a_t = 0
        self.delta_e_t = 0
        self.delta_r_t = 0
        self.delta_F_t = 0
        self.Va_t = 0
        self.Vg_t = 0
        self.alpha_t = 0

        self.time_e = 0
        self.bx_e = 0
        self.by_e = 0
        self.bz_e = 0
        self.ax_e = 0
        self.ay_e = 0
        self.az_e = 0
        self.phi_e = 0
        self.theta_e = 0
        self.psi_e = 0
        self.p_e = 0
        self.q_e = 0
        self.r_e = 0
        self.delta_a_e = 0
        self.delta_e_e = 0
        self.delta_r_e = 0
        self.delta_F_e = 0
        self.Va_e = 0
        self.Vg_e = 0
        self.alpha_e = 0
        # EKF attitude estimates
        self.phi_e2 = 0
        self.theta_e2 = 0
        self.psi_e2 = 0

        # truth/estimate storage lists
        self.estimates = []
        self.estimates2 = []
        self.truths = []

        # plot list
        self.p_list = [self.p_bx, self.p_by, self.p_bz, self.p_phi, self.p_theta, self.p_psi, self.p_ax, self.p_ay, self.p_az, self.p_p, self.p_q, self.p_r, self.p_delta_a, self.p_delta_e, self.p_delta_r, self.p_delta_F, self.p_Va, self.p_Vg, self.p_alpha]
        for plot in self.p_list:
            state = plot.getViewBox().getState()
            state["autoVisibleOnly"] = [False, True]
            plot.getViewBox().setState(state)

        # curve lists
        self.c_list_t = [self.c_bx_t, self.c_by_t, self.c_bz_t, self.c_ax_t, self.c_ay_t, self.c_az_t, self.c_phi_t, self.c_theta_t, self.c_psi_t, self.c_p_t, self.c_q_t, self.c_r_t, self.c_delta_a_t, self.c_delta_e_t, self.c_delta_r_t, self.c_delta_F_t, self.c_Va_t, self.c_Vg_t, self.c_alpha_t]
        self.c_list_e = [self.c_bx_e, self.c_by_e, self.c_bz_e, self.c_ax_e, self.c_ay_e, self.c_az_e, self.c_phi_e, self.c_theta_e, self.c_psi_e, self.c_p_e, self.c_q_e, self.c_r_e, self.c_delta_a_e, self.c_delta_e_e, self.c_delta_r_e, self.c_delta_F_e, self.c_Va_e, self.c_Vg_e, self.c_alpha_e]
        self.c_list_e2 = [self.c_phi_e2, self.c_theta_e2, self.c_psi_e2]

    # method for updating each states
    def update(self):
        # TEST:
        # self.bx_e = self.phi_e - self.phi_t
        # self.by_e = self.theta_e - self.theta_t
        # self.bz_e = self.psi_e - self.psi_t

        if self.prev_time != self.time_t:
            # pack stored data into lists
            self.truths.append([self.time_t, self.bx_t, self.by_t, self.bz_t, self.phi_t, self.theta_t, self.psi_t, self.ax_t, self.ay_t, self.az_t, self.p_t, self.q_t, self.r_t, self.delta_a_t, self.delta_e_t, self.delta_r_t, self.delta_F_t, self.Va_t, self.Vg_t, self.alpha_t])
            self.estimates.append([self.time_e, self.bx_e, self.by_e, self.bz_e, self.phi_e, self.theta_e, self.psi_e, self.ax_e, self.ay_e, self.az_e, self.p_e, self.q_e, self.r_e, self.delta_a_e, self.delta_e_e, self.delta_r_e, self.delta_F_e, self.Va_e, self.Vg_e, self.alpha_e])
            self.estimates2.append([self.time_e, self.phi_e2, self.theta_e2, self.psi_e2])

            # self.characterizeAngleError()

            # self.truths = self.limitArrayLength(self.truths, self.curve_length)

            # self.estimates = self.limitArrayLength(self.estimates, self.curve_length)

            # stack the data lists
            truths_array = np.vstack(self.truths)
            time_t_array = truths_array[:,0]

            estimates_array = np.vstack(self.estimates)
            estimates2_array = np.vstack(self.estimates2)
            time_e_array = estimates_array[:,0]

            # set the truth states
            for i in range(0,len(self.c_list_t)):
    	        self.c_list_t[i].setData(time_t_array, truths_array[:,i+1], pen=(255,0,0))

            # set the estimated states
            for i in range(0,len(self.c_list_e)):
                self.c_list_e[i].setData(time_e_array, estimates_array[:,i+1], pen=(0,255,0))
            for i in range(0,len(self.c_list_e2)):
                self.c_list_e2[i].setData(time_e_array, estimates2_array[:,i+1], pen=(255,255,0))

            x_min = util.sat(self.time_t - self.time_window, 0, self.time_t - self.time_window)
            x_max = self.time_t
            for i in range(0,len(self.p_list)):
                self.p_list[i].setXRange(x_min, x_max)
                self.p_list[i].enableAutoRange(axis=ViewBox.YAxis)

        self.prev_time = self.time_t
        # update the plotted data
        self.app.processEvents()


    def truthCallback(self, msg):
        if type(msg) == State:

            self.Vg_t = msg.Vg
            self.Va_t = msg.Va

            # unpack angles and angular velocities
            self.phi_t = msg.phi * 180.0 / math.pi
            self.theta_t = msg.theta * 180.0 / math.pi
            self.psi_t = msg.psi * 180.0 / math.pi
            self.p_t = msg.p
            self.q_t = msg.q
            self.r_t = msg.r
            self.alpha_t = msg.alpha * 180.0 / math.pi

        elif type(msg) == Odometry:
            quat = msg.pose.pose.orientation
            q = (quat.x, quat.y, quat.z, quat.w)
            euler = tf.transformations.euler_from_quaternion(q)

            self.phi_t = euler[0] * 180 / math.pi
            self.theta_t = euler[1] * 180 / math.pi
            self.psi_t = euler[2] * 180 / math.pi

            omega = msg.twist.twist.angular

            self.p_t = omega.x
            self.q_t = omega.y
            self.r_t = omega.z


        # unpack time
        now = msg.header.stamp.to_sec()
        if self.init_time == True:
        	# self.time0 = rospy.get_rostime().to_sec()
            self.time0 = now
            # zero out the estimate time reference
            self.time_e0 += self.time_e
            self.init_time = False
        self.time_t = now - self.time0

        # Error characterization
        # self.phi_error.append((self.phi_e - self.phi_t))
        # self.theta_error.append((self.theta_e - self.theta_t))
        # # self.psi_error.append((self.psi_e - self.psi_t))
        #
        # self.mekf_phi_error.append((self.phi_e2 - self.phi_t))
        # self.mekf_theta_error.append((self.theta_e2 - self.theta_t))
        # # self.mekf_psi_error.append((self.psi_e2 - self.psi_t))

        if self.time_t < 15.0:
            return # Don't start collecting data until after takeoff

        self.data_cnt += 1
        if self.data_cnt >= 2:
            phi_x = util.angleDiff(self.phi_e, self.phi_t, degrees=True)
            [self.phi_mean, self.phi_mean_sq, self.phi_M2, self.phi_variance] = self.getRunningStats(phi_x, self.data_cnt, self.phi_mean, self.phi_mean_sq, self.phi_M2)
            theta_x = util.angleDiff(self.theta_e, self.theta_t, degrees=True)
            [self.theta_mean, self.theta_mean_sq, self.theta_M2, self.theta_variance] = self.getRunningStats(theta_x, self.data_cnt, self.theta_mean, self.theta_mean_sq, self.theta_M2)
            psi_x = util.angleDiff(self.psi_e, self.psi_t, degrees=True)
            [self.psi_mean, self.psi_mean_sq, self.psi_M2, self.psi_variance] = self.getRunningStats(psi_x, self.data_cnt, self.psi_mean, self.psi_mean_sq, self.psi_M2)
            mekf_phi_x = util.angleDiff(self.phi_e2, self.phi_t, degrees=True)
            [self.mekf_phi_mean, self.mekf_phi_mean_sq, self.mekf_phi_M2, self.mekf_phi_variance] = self.getRunningStats(mekf_phi_x, self.data_cnt, self.mekf_phi_mean, self.mekf_phi_mean_sq, self.mekf_phi_M2)
            mekf_theta_x = util.angleDiff(self.theta_e2, self.theta_t, degrees=True)
            [self.mekf_theta_mean, self.mekf_theta_mean_sq, self.mekf_theta_M2, self.mekf_theta_variance] = self.getRunningStats(mekf_theta_x, self.data_cnt, self.mekf_theta_mean, self.mekf_theta_mean_sq, self.mekf_theta_M2)
            mekf_psi_x = util.angleDiff(self.psi_e2, self.psi_t, degrees=True)
            [self.mekf_psi_mean, self.mekf_psi_mean_sq, self.mekf_psi_M2, self.mekf_psi_variance] = self.getRunningStats(mekf_psi_x, self.data_cnt, self.mekf_psi_mean, self.mekf_psi_mean_sq, self.mekf_psi_M2)


        # if self.data_cnt % self.error_block_size == 0:
        #     # phi_mse = np.mean(np.multiply(self.phi_error, self.phi_error))
        #     # phi_avg_error = np.mean(self.phi_error)
        #     # theta_mse = np.mean(np.multiply(self.theta_error, self.theta_error))
        #     # theta_avg_error = np.mean(self.theta_error)
        #     # phi_error_std_dev = np.std(self.phi_error)
        #     # theta_error_std_dev = np.std(self.theta_error)
        #     # mekf_phi_mse = np.mean(np.multiply(self.mekf_phi_error, self.mekf_phi_error))
        #     # mekf_phi_avg_error = np.mean(self.mekf_phi_error)
        #     # mekf_theta_mse = np.mean(np.multiply(self.mekf_theta_error, self.mekf_theta_error))
        #     # mekf_theta_avg_error = np.mean(self.mekf_theta_error)
        #     # mekf_phi_error_std_dev = np.std(self.mekf_phi_error)
        #     # mekf_theta_error_std_dev = np.std(self.mekf_theta_error)
        #     # psi_mse = np.mean(self.psi_error)
        #     print(" =================== Angle Error Characterization @ t = {0} =================== ".format(self.time_t))
        #     print("\t\t<<<<<<<< ROSplane EKF >>>>>>>> ")
        #     print("\t\t  phi (deg):    Avg. Err. = {0}, \t MSE = {1}, \t Var. = {2}".format(self.phi_mean, self.phi_mean_sq, self.phi_variance))
        #     print("\t\t  theta (deg):  Avg. Err. = {0}, \t MSE = {1}, \t Var. = {2}".format(self.theta_mean, self.theta_mean_sq, self.theta_variance))
        #     print("\t\t  psi (deg):    Avg. Err. = {0}, \t MSE = {1}, \t Var. = {2}".format(self.psi_mean, self.psi_mean_sq, self.psi_variance))
        #     print("\t\t<<<<<<<< MEKF >>>>>>>> ")
        #     print("\t\t  phi (deg):    Avg. Err. = {0}, \t MSE = {1}, \t Var. = {2}".format(self.mekf_phi_mean, self.mekf_phi_mean_sq, self.mekf_phi_variance))
        #     print("\t\t  theta (deg):  Avg. Err. = {0}, \t MSE = {1}, \t Var. = {2}".format(self.mekf_theta_mean, self.mekf_theta_mean_sq, self.mekf_theta_variance))
        #     print("\t\t  psi (deg):    Avg. Err. = {0}, \t MSE = {1}, \t Var. = {2}".format(self.mekf_psi_mean, self.mekf_psi_mean_sq, self.mekf_psi_variance))

        # # Alpha system ID
        # dt = self.time_t = self.prev_time
        # self.prev_time = self.time_e
        #
        # alpha = msg.alpha
        # self.alphadot = util.tustinDerivative(self.alphadot, alpha, self.prev_alpha, dt, self.tau)
        # self.prev_alpha = alpha
        #
        # Va = util.sat(self.Va_t, 0.001, self.Va_t)
        # alphadot_e = -1.0*(self.c0/Va)*alpha + self.q_t + self.alpha0
        # alpha_error = self.alphadot - alphadot_e
        #
        # if alpha > 0:
        #     self.c0 += self.c0_kp * alpha_error

        # self.alpha0 += self.alpha0_kp*alpha_error

        # print(" <<<< Alpha testing >>>> ")
        # print("alphadot_t: {0}".format(self.alphadot))
        # print("c0: {0}".format(self.c0))
        # print("airspeed: {0}".format(self.Va_t))
        # print("alpha: {0}".format(alpha))
        # print("thetadot: {0}".format(self.q_t))
        # print("alpha0: {0}".format(self.alpha0))
        # print("alphadot_e: {0}".format(alphadot_e))
        # print("error: {0}".format(alpha_error))
        # print("")

    # Statistics function that doesn't require storage of the data points
    # Returns updated values for n, mean, M2 (used to compute variance), and the current variance
    def getRunningStats(self, x, n, mean, mean_sq, M2):
        # Update mean squared error data
        x_sq = x**2
        delta_sq = x_sq - mean_sq
        mean_sq += delta_sq/n

        delta = x - mean
        mean += delta/n
        delta2 = x - mean
        M2 += delta*delta2
        var = M2 / (n - 1)
        return [mean, mean_sq, M2, var]

    def estimateCallback(self, msg):
        # unpack positions and linear velocities
        # self.bx_e = msg.position[0]
        # self.by_e = msg.position[1]
        # self.bz_e = -msg.position[2]

        # Not part of the message for now
        # self.wn_e = msg.wn
        # self.we_e = msg.we
        # self.chi_e = msg.chi

        self.Vg_e = msg.Vg
        self.Va_e = msg.Va

        # orientation in quaternion form
        # quaternion = msg.quat
        # Use ROS tf to convert to Euler angles from quaternion
        # euler = tf.transformations.euler_from_quaternion(quaternion)

        # unpack angles and angular velocities
        # self.phi_e = euler[0]
        # self.theta_e = euler[1]
        # self.psi_e = euler[2]
        self.phi_e = msg.phi * 180.0 / math.pi
        self.theta_e = msg.theta * 180.0 / math.pi
        self.psi_e = util.angleMod(msg.psi, max_angle=math.pi) * 180.0 / math.pi
        # self.p_e = msg.p
        # self.q_e = msg.q
        # self.r_e = msg.r


    def selfPoseCallback(self, msg):
        # Extract attitude estimates
        quat = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        [phi, theta, psi] = tf.transformations.euler_from_quaternion(quat)
        self.phi_e2 = phi * 180.0 / math.pi
        self.theta_e2 = theta * 180.0 / math.pi
        self.psi_e2 = psi * 180.0 / math.pi

        # Extract gyro estimates
        self.p_e = msg.angular_velocity.x
        self.q_e = msg.angular_velocity.y
        self.r_e = msg.angular_velocity.z

        # unpack time
        now = msg.header.stamp.to_sec()
        if self.init_time_e == True:
            self.time_e0 = now
            self.init_time_e = False
            # zero out the truth time reference
            self.time0 += self.time_t

        self.time_e = now - self.time_e0


    def attitudeEstimateCallback(self, msg):
        self.phi_e2 = msg.x * 180.0 / math.pi
        self.theta_e2 = msg.y * 180.0 / math.pi
        self.psi_e2 = msg.z * 180.0 / math.pi

    def commandCallback(self, msg):
        self.delta_a_t = msg.x
        self.delta_e_t = msg.y
        self.delta_r_t = msg.z
        self.delta_F_t = msg.F


    def biasEstimateCallback(self, msg):
        self.bx_e = msg.x * 180.0 / math.pi
        self.by_e = msg.y * 180.0 / math.pi
        self.bz_e = msg.z * 180.0 / math.pi
        # self.characterizeBiasError()

    def gyroEstimateCallback(self, msg):
        self.p_e = msg.x
        self.q_e = msg.y
        self.r_e = msg.z

    def biasTruthCallback(self, msg):
        self.bx_t = msg.vector.x * 180.0 / math.pi
        self.by_t = msg.vector.y * 180.0 / math.pi
        self.bz_t = msg.vector.z * 180.0 / math.pi

    def imuCallback(self, msg):
        self.p_imu = msg.angular_velocity.x
        self.q_imu = msg.angular_velocity.y
        self.r_imu = msg.angular_velocity.z

        self.ax_t = msg.linear_acceleration.x
        self.ay_t = msg.linear_acceleration.y
        self.az_t = msg.linear_acceleration.z

    def alphaCallback(self, msg):
        self.alpha_e = msg.data * 180.0 / math.pi

    def characterizeBiasError(self):
        truths_array = np.vstack(self.truths)
        estimates_array = np.vstack(self.estimates)

        bias_truths = truths_array[:,1:4]
        bias_estimates = estimates_array[:,1:4]
        bias_errors = bias_estimates - bias_truths # [ex, ey, ez] (N x 3 matrix)
        average_error = np.mean(bias_errors, axis=0)
        std_dev = np.std(bias_errors, axis=0)
        print(" ============= BIAS RESULTS ============= ")
        print("Average error: {0}".format(average_error))
        print("Standard dev: {0}".format(std_dev))

    def characterizeAngleError(self):
        truths_array = np.vstack(self.truths)
        estimates_array = np.vstack(self.estimates)
        estimates2_array = np.vstack(self.estimates2)

        angle_truths = truths_array[:,4:7]
        angle_estimates = estimates_array[:,4:7]
        angle_estimates2 = estimates2_array[:,1:4]

        angle_errors = angle_estimates - angle_truths # [ex, ey, ez] (N x 3 matrix
        average_error = np.mean(angle_errors, axis=0)
        std_dev = np.std(angle_errors, axis=0)

        # ROS Plane EKF results
        angle_errors2 = angle_estimates2 - angle_truths # [ex, ey, ez] (N x 3 matrix
        average_error2 = np.mean(angle_errors2, axis=0)
        std_dev2 = np.std(angle_errors2, axis=0)
        print(" ============= ANGLE RESULTS ============= ")
        print(" ---- Mahony Filter ---- ")
        print("Average error: {0}".format(average_error))
        print("Standard dev: {0}".format(std_dev))
        print(" ---- ROS Plane Filter ---- ")
        print("Average error: {0}".format(average_error2))
        print("Standard dev: {0}".format(std_dev2))

    def limitArrayLength(self, array, length):
        size = np.size(array, 0)
        size_diff = size - length
        for i in range(1, size_diff):
            array.pop(0)
        return array


################################################################################
################################################################################
################################################################################


def main():
    # initialize node
    rospy.init_node('state_plotter', anonymous=True)

    # initialize plotter class
    plotter = Plotter()

    # listen for messages and plot
    while not rospy.is_shutdown():
        try:
            # plot the local positions of each vehicle
            plotter.update()

            # let it rest a bit
            time.sleep(0.001)
        except rospy.ROSInterruptException:
            print "exiting...."
            return

if __name__ == '__main__':
    main()
