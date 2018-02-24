from IPython.core.debugger import set_trace
import cv2
import numpy as np


############ Kalman Filter Wrapper ############
class Filter:
    CONST_VEL = 1
    CONST_ACCEL = 2
    CONST_JERK = 3
    def __init__(self, model_type=CONST_VEL, dt=1/30.0):
        self.dt = dt
        self.model_type = model_type

        self.meas_dim = 4
        self.sigma_R  = 20.0
        if model_type == self.CONST_VEL:
            self.state_dim = 4
            self.sigma_Q = 1.0
        elif model_type == self.CONST_ACCEL:
            self.state_dim = 6
            self.sigma_Q = 0.01
        elif model_type == self.CONST_JERK:
            self.state_dim = 8
            self.sigma_Q = 0.005

        data_type = np.float32
        self.kalman = cv2.KalmanFilter(self.state_dim,self.meas_dim)

        # Matrix definitions
        self.kalman.measurementMatrix   = self.get_C_matrix().astype(data_type)
        self.kalman.transitionMatrix    = self.get_A_matrix().astype(data_type)
        self.kalman.processNoiseCov     = self.get_Q_matrix().astype(data_type)
        self.kalman.measurementNoiseCov = self.sigma_R*np.eye(self.meas_dim).astype(data_type)

        # Initial conditions
        P_init = self.sigma_R # We use the first measurement to initialize the state
        self.kalman.errorCovPost = P_init*np.eye(self.state_dim).astype(data_type)
        self.kalman.errorCovPre = self.kalman.errorCovPost
        self.state_init = False

    def get_A_matrix(self):
        n = int(self.state_dim/2)
        A = np.eye(n)
        for row in range(0,n):
            for col in range(row+1,n):
                A[row,col] = self.dt**(col-row)
        return np.kron(A, np.eye(2))

    def get_Q_matrix(self):
        dt = self.dt
        if self.model_type == self.CONST_VEL:
            Q = np.array([[1.0/3*dt**3, 1.0/2*dt**2],
                          [1.0/2*dt**2,       dt   ]])
        elif self.model_type == self.CONST_ACCEL:
            Q = np.array([[1.0/20*dt**5, 1.0/8*dt**4, 1.0/6*dt**3],
                          [1.0/8 *dt**4, 1.0/3*dt**3, 1.0/2*dt**2],
                          [1.0/6 *dt**3, 1.0/2*dt**2,       dt   ]])
        elif self.model_type == self.CONST_JERK:
            Q = np.array([[1.0/252*dt**7, 1.0/72*dt**6, 1.0/30*dt**5, 1.0/24*dt**4],
                          [1.0/72 *dt**6, 1.0/20*dt**5, 1.0/8 *dt**4, 1.0/6 *dt**3],
                          [1.0/30 *dt**5, 1.0/8 *dt**4, 1.0/3 *dt**3, 1.0/2 *dt**2],
                          [1.0/24 *dt**4, 1.0/6 *dt**3, 1.0/2 *dt**2,        dt   ]])
        return self.sigma_Q*np.kron(Q, np.eye(2))

    def get_C_matrix(self):
        I = np.eye(self.meas_dim)
        zeros = np.zeros((self.meas_dim, self.state_dim - self.meas_dim))
        C = np.hstack((I,zeros))

        return C

    def get_S_matrix(self):
        P = self.kalman.errorCovPre
        C = self.kalman.measurementMatrix
        R = self.kalman.measurementNoiseCov

        S = np.matmul(np.matmul(C,P),C.transpose()) + R
        return S

    def predict(self):
        return self.kalman.predict()

    def correct(self, meas):
        # Only run the correction if there is a measurement
        if len(meas) == 0:
            return

        # Initialize state with first measurement
        if not self.state_init:
            zeros = np.zeros((self.state_dim - self.meas_dim,1))
            init_state = np.vstack((meas,zeros))
            self.kalman.statePost = init_state.astype(np.float32)
            self.kalman.statePre = init_state.astype(np.float32)
            self.state_init = True

        return self.kalman.correct(meas.astype(np.float32))
