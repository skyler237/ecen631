import numpy as np
import pygame
from pygame.locals import *
import math
import curses

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors


TIME_OUT_VAL = 1000

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
# ROLL_RIGHT  = ord('d')
# ROLL_LEFT   = ord('a')
# PITCH_UP    = ord('w')
# PITCH_DOWN  = ord('s')
# YAW_LEFT    = ord('j')
# YAW_RIGHT   = ord('l')
# ALT_UP      = ord('i')
# ALT_DOWN    = ord('k')
# QUIT        = 27


# This is a basic example of how to use the UAV agent
def uav_example():
    (screen, font) = initialize_pygame()
    text = "nothing yet..."
    # pygame.init()
    # # pygame.event.set_grab(True)

    # stdscr = curses.initscr()
    # curses.cbreak()
    # stdscr.keypad(1)

    env = Holodeck.make("UrbanCity")

    # Initialize watchdog timer`
    timeout_cnt = TIME_OUT_VAL

    # Initialize altitude to zero
    alt_c = 0
    speed_val = 0.0

    # Rate parameters
    roll_min = math.radians(10)
    roll_max = math.radians(45)
    pitch_min = math.radians(10)
    pitch_max = math.radians(45)
    yawrate_min = math.radians(30)
    yawrate_max = math.radians(90)
    altrate_min = 0.1
    altrate_max = 0.5
    speed_min = 0.0
    speed_max = 1.0
    speed_rate = 0.1

    # Loop forever
    while True:
        timeout_cnt -= 1
        # Default all angles/rates to zero at each time step
        roll_c = 0
        pitch_c = 0
        yawrate_c = 0
        # Check keyboard input

        pygame.event.pump()
        keys=pygame.key.get_pressed()
        # key = stdscr.getch()

        # Update control values
        if keys[K_ESCAPE] or timeout_cnt < 0:
            break # Quit the program
        if keys[ROLL_RIGHT]:
            timeout_cnt = TIME_OUT_VAL
            roll_c = -(roll_min + (roll_max - roll_min)*speed_val)
            text = "ROLL_RIGHT"
        if keys[ROLL_LEFT]:
            timeout_cnt = TIME_OUT_VAL
            roll_c = (roll_min + (roll_max - roll_min)*speed_val)
            text = "ROLL_LEFT"
        if keys[PITCH_UP]:
            timeout_cnt = TIME_OUT_VAL
            pitch_c = (pitch_min + (pitch_max - pitch_min)*speed_val)
            text = "PITCH_UP"
        if keys[PITCH_DOWN]:
            timeout_cnt = TIME_OUT_VAL
            pitch_c = -(pitch_min + (pitch_max - pitch_min)*speed_val)
            text = "PITCH_DOWN"
        if keys[YAW_LEFT]:
            timeout_cnt = TIME_OUT_VAL
            yawrate_c = (yawrate_min + (yawrate_max - yawrate_min)*speed_val)
            text = "YAW_LEFT"
        if keys[YAW_RIGHT]:
            timeout_cnt = TIME_OUT_VAL
            yawrate_c = -(yawrate_min + (yawrate_max - yawrate_min)*speed_val)
            text = "YAW_RIGHT"
        if keys[ALT_UP]:
            timeout_cnt = TIME_OUT_VAL
            alt_c += (altrate_min + (altrate_max - altrate_min)*speed_val)
            text = "Altitude raised to {0}".format(alt_c)
        if keys[ALT_DOWN]:
            timeout_cnt = TIME_OUT_VAL
            alt_c -= max(((altrate_min + (altrate_max - altrate_min)*speed_val), 0))
            text = "Altitude lowered to {0}".format(alt_c)
        if keys[SPEED_UP]:
            timeout_cnt = TIME_OUT_VAL
            speed_val += speed_rate
            speed_val = min(speed_val, speed_max)
            text = "Speed raised to {0}".format(speed_val)
        if keys[SPEED_DOWN]:
            timeout_cnt = TIME_OUT_VAL
            speed_val -= speed_rate
            speed_val = max(speed_val, speed_min)
            text = "Speed lowered to {0}".format(speed_val)
        # if key == K_ESCAPE or timeout_cnt < 0:
        #     break # Quit the program
        # if key == ROLL_RIGHT:
        #     timeout_cnt = TIME_OUT_VAL
        #     roll_c = roll_min
        # if key == ROLL_LEFT:
        #     timeout_cnt = TIME_OUT_VAL
        #     roll_c = -roll_min
        # if key == PITCH_UP:
        #     timeout_cnt = TIME_OUT_VAL
        #     pitch_c = pitch_min
        # if key == PITCH_DOWN:
        #     timeout_cnt = TIME_OUT_VAL
        #     pitch_c = -pitch_min
        # if key == YAW_LEFT:
        #     timeout_cnt = TIME_OUT_VAL
        #     yawrate_c = -yawrate_min
        # if key == YAW_RIGHT:
        #     timeout_cnt = TIME_OUT_VAL
        #     yawrate_c = yawrate_min
        # if key == ALT_UP:
        #     timeout_cnt = TIME_OUT_VAL
        #     alt_c += altrate_min
        # if key == ALT_DOWN:
        #     timeout_cnt = TIME_OUT_VAL
        #     alt_c -= max(altrate_min, 0)

        # text = "key pressed!"
        # screen.fill((0,0,0))
        # block = font.render(text, True, (255,255,255))
        # rect = block.get_rect()
        # rect.center = screen.get_rect().center
        # screen.blit(block, rect)
        # pygame.display.flip()


        # Construct command
        command = np.array([roll_c, pitch_c, yawrate_c, alt_c])

        # Step simulator
        state, reward, terminal, info = env.step(command)

        # Could access sensor data here

SURFACE_WIDTH = 640
SURFACE_HEIGHT = 480

KEY_BOX_WIDTH = 80
KEY_LINE_WIDTH = 3
KEY_VALUE_SIZE = 30
KEY_TEXT_SIZE = 15
KEY_PADDING = 10
KEY_SIZE = (KEY_BOX_WIDTH, KEY_BOX_WIDTH)
KEY_DEFAULT_COLOR = (255,255,255) # White
KEY_PRESSED_COLOR = (0,0,255) # Red

SIDE_PADDING = 30
CENTER_PADDING = 30
TOP_Y = 120

A_COORD = (SIDE_PADDING, TOP_Y + KEY_BOX_WIDTH + KEY_PADDING)
W_COORD = (SIDE_PADDING + KEY_BOX_WIDTH + KEY_PADDING, TOP_Y)
S_COORD = (SIDE_PADDING + KEY_BOX_WIDTH + KEY_PADDING, TOP_Y + KEY_BOX_WIDTH + KEY_PADDING)
D_COORD = (SIDE_PADDING + 2*(KEY_BOX_WIDTH + KEY_PADDING), TOP_Y + KEY_BOX_WIDTH + KEY_PADDING)

RIGHT_COORD = (SURFACE_WIDTH - KEY_BOX_WIDTH - SIDE_PADDING, TOP_Y + KEY_BOX_WIDTH + KEY_PADDING)
UP_COORD = (SURFACE_WIDTH - KEY_BOX_WIDTH - (SIDE_PADDING + KEY_BOX_WIDTH + KEY_PADDING), TOP_Y)
DOWN_COORD = (SURFACE_WIDTH - KEY_BOX_WIDTH - (SIDE_PADDING + KEY_BOX_WIDTH + KEY_PADDING), TOP_Y + KEY_BOX_WIDTH + KEY_PADDING)
LEFT_COORD = (SURFACE_WIDTH - KEY_BOX_WIDTH - (SIDE_PADDING + 2*(KEY_BOX_WIDTH + KEY_PADDING)), TOP_Y + KEY_BOX_WIDTH + KEY_PADDING)


def initialize_pygame():
    pygame.init()
    screen = pygame.display.set_mode( (SURFACE_WIDTH,SURFACE_HEIGHT) )
    pygame.display.set_caption('Holodeck UAV Teleop')

    a_rect = Rect(A_COORD, KEY_SIZE)
    w_rect = Rect(W_COORD, KEY_SIZE)
    s_rect = Rect(S_COORD, KEY_SIZE)
    d_rect = Rect(D_COORD, KEY_SIZE)

    up_rect = Rect(UP_COORD, KEY_SIZE)
    down_rect = Rect(DOWN_COORD, KEY_SIZE)
    left_rect = Rect(LEFT_COORD, KEY_SIZE)
    right_rect = Rect(RIGHT_COORD, KEY_SIZE)

    pygame.draw.rect(screen, KEY_DEFAULT_COLOR, s_rect, KEY_LINE_WIDTH)
    pygame.draw.rect(screen, KEY_DEFAULT_COLOR, d_rect, KEY_LINE_WIDTH)
    pygame.draw.rect(screen, KEY_DEFAULT_COLOR, w_rect, KEY_LINE_WIDTH)
    pygame.draw.rect(screen, KEY_DEFAULT_COLOR, a_rect, KEY_LINE_WIDTH)

    pygame.draw.rect(screen, KEY_DEFAULT_COLOR, up_rect, KEY_LINE_WIDTH)
    pygame.draw.rect(screen, KEY_DEFAULT_COLOR, down_rect, KEY_LINE_WIDTH)
    pygame.draw.rect(screen, KEY_DEFAULT_COLOR, left_rect, KEY_LINE_WIDTH)
    pygame.draw.rect(screen, KEY_DEFAULT_COLOR, right_rect, KEY_LINE_WIDTH)
    pygame.display.flip()

    font = pygame.font.Font(None, 50)
    return (screen, font)



if __name__ == "__main__":
    uav_example()
    print("Finished")
