import numpy as np
import math
import scipy.io as sio # For exporting to matlab
import pygame
from pygame.locals import *

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
from uav_state_plotter import Plotter

# Command key mappings
ROLL_RIGHT  = K_d
ROLL_LEFT   = K_a
PITCH_UP    = K_w
PITCH_DOWN  = K_s
YAW_LEFT    = K_LEFT
YAW_RIGHT   = K_RIGHT
ALT_UP      = K_UP
ALT_DOWN    = K_DOWN
SPEED_UP    = K_e
SPEED_DOWN  = K_q
QUIT = K_ESCAPE
PAUSE = K_SPACE

class UAVSim():
    def __init__(self, world):
        ### Parameters
        # Default command
        self.roll_c = 0
        self.pitch_c = 0
        self.yawrate_c = 0
        self.alt_c = 0
        self.command = np.array([self.roll_c, self.pitch_c, self.yawrate_c, self.alt_c])

        # Rate parameters
        self.roll_min = math.radians(10)
        self.roll_max = math.radians(45)
        self.pitch_min = math.radians(10)
        self.pitch_max = math.radians(45)
        self.yawrate_min = math.radians(30)
        self.yawrate_max = math.radians(180)
        self.altrate_min = 0.1
        self.altrate_max = 0.5
        self.speed_min = 0.0
        self.speed_max = 1.0
        self.speed_rate = 0.1
        self.speed_val = 0

        # Teleop
        self.using_teleop = False
        self.teleop_text = "Click here to use teleop"

        # Simulation return variables
        self.sim_state = 0
        self.sim_reward = 0
        self.sim_terminal = 0
        self.sim_info = 0
        self.sim_step = 0
        self.dt = 0.01

        # Data saving options
        self.saving_state = False
        self.sim_state_list = []
        self.sim_step_list = []
        self.state_file = 'states.mat'

        self.plotting_states = False

        # Initialize world
        self.env = Holodeck.make(world)
        self.paused = False

    ######## Plotting Functions ########
    def init_plots(self):
        self.plotting_states = True
        self.plotter = Plotter()

    ######## Teleop Functions ########
    def init_teleop(self):
        self.using_teleop = True
        pygame.init()
        SURFACE_WIDTH = 640
        SURFACE_HEIGHT = 480
        self.teleop_screen = pygame.display.set_mode( (SURFACE_WIDTH,SURFACE_HEIGHT) )
        pygame.display.set_caption('Holodeck UAV Teleop')

        self.teleop_font = pygame.font.Font(None, 50)

    def update_teleop_display(self):
        self.teleop_screen.fill((0,0,0))
        block = self.teleop_font.render(self.teleop_text, True, (255,255,255))
        rect = block.get_rect()
        rect.center = self.teleop_screen.get_rect().center
        self.teleop_screen.blit(block, rect)
        pygame.display.flip()

    def get_teleop_command(self):
        pygame.event.pump()
        keys=pygame.key.get_pressed()

        # Default all angles/rates to zero at each time step
        self.roll_c = 0
        self.pitch_c = 0
        self.yawrate_c = 0

        # Update control values
        if keys[QUIT]:
            # return False # Quit the program
            self.exit_sim()
        if keys[PAUSE]:
            self.paused = self.paused ^ True
            self.teleop_text = "Simulation paused"
            # TODO: add debouncing timer
        if self.paused:
            return
        if keys[ROLL_RIGHT]:
            self.roll_c = -(self.roll_min + (self.roll_max - self.roll_min)*self.speed_val)
            self.teleop_text = "ROLL_RIGHT"
        if keys[ROLL_LEFT]:
            self.roll_c = (self.roll_min + (self.roll_max - self.roll_min)*self.speed_val)
            self.teleop_text = "ROLL_LEFT"
        if keys[PITCH_UP]:
            self.pitch_c = (self.pitch_min + (self.pitch_max - self.pitch_min)*self.speed_val)
            self.teleop_text = "PITCH_UP"
        if keys[PITCH_DOWN]:
            self.pitch_c = -(self.pitch_min + (self.pitch_max - self.pitch_min)*self.speed_val)
            self.teleop_text = "PITCH_DOWN"
        if keys[YAW_LEFT]:
            self.yawrate_c = (self.yawrate_min + (self.yawrate_max - self.yawrate_min)*self.speed_val)
            self.teleop_text = "YAW_LEFT"
        if keys[YAW_RIGHT]:
            self.yawrate_c = -(self.yawrate_min + (self.yawrate_max - self.yawrate_min)*self.speed_val)
            self.teleop_text = "YAW_RIGHT"
        if keys[ALT_UP]:
            self.alt_c += (self.altrate_min + (self.altrate_max - self.altrate_min)*self.speed_val)
            self.teleop_text = "Altitude raised to {0}".format(self.alt_c)
        if keys[ALT_DOWN]:
            self.alt_c -= max(((self.altrate_min + (self.altrate_max - self.altrate_min)*self.speed_val), 0))
            self.teleop_text = "Altitude lowered to {0}".format(self.alt_c)
        if keys[SPEED_UP]:
            self.speed_val += self.speed_rate
            self.speed_val = min(self.speed_val, self.speed_max)
            self.teleop_text = "Speed raised to {0}".format(self.speed_val)
        if keys[SPEED_DOWN]:
            self.speed_val -= self.speed_rate
            self.speed_val = max(self.speed_val, self.speed_min)
            self.teleop_text = "Speed lowered to {0}".format(self.speed_val)

        self.command = np.array([self.roll_c, self.pitch_c, self.yawrate_c, self.alt_c])

        return True

    ######## Data access ########
    def get_state(self):
        return self.sim_state

    def set_state_file(self, state_file):
        self.saving_state = True
        self.state_file = state_file

    def write_state(self):
        sio.savemat(self.state_file, {'states':np.ravel(self.sim_state_list), 't':np.ravel(self.sim_step_list)})

    def get_camera(self):
        return np.asarray(self.sim_state[Sensors.PRIMARY_PLAYER_CAMERA])

    def get_position(self):
        return self.sim_state[Sensors.LOCATION_SENSOR]

    def get_velocity(self):
        return self.sim_state[Sensors.VELOCITY_SENSOR]

    def get_imu(self):
        return self.sim_state[Sensors.IMU_SENSOR]

    def get_orientation(self):
        return self.sim_state[Sensors.ORIENTATION_SENSOR]

    def step_sim(self):
        if self.using_teleop:
            self.get_teleop_command()
            self.update_teleop_display()

        # Step holodeck simulator
        if not self.paused:
            self.sim_step += 1
            self.sim_state, self.sim_reward, self.sim_terminal, self.sim_info = self.env.step(self.command)
            if self.saving_state:
                self.sim_state_list.append(self.sim_state)
                self.sim_step_list.append(self.sim_step)
                self.write_state() # REVIEW: Is this too frequent?
            if self.plotting_states:
                self.plotter.update_states(self.sim_state, self.sim_step*self.dt)
                self.plotter.update_commands(self.command)
                self.plotter.update_plots()

    def exit_sim(self):
        if self.saving_state:
            self.write_state()
        quit()
