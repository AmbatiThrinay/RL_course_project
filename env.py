""" Acrobot ENV"""

import numpy as np
import cv2

# supressing the pygame's hello message
import os, sys
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
# <------ for headless execution(server) --------->
# set SDL to use the dummy NULL video driver, 
# so it doesn't need a windowing system.
# os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame


class Acrobot:
    pygame.init()
    pygame.display.set_caption("Acrobot Environment")
    FONT = pygame.font.SysFont('ComicNeue-Regular.ttf',21)
    """
    ### Description
    The Acrobot environment is based on Sutton and Barto's book.

    ### Action Space
    | Num | Action                                | Unit         |
    |-----|---------------------------------------|--------------|
    | 0   | apply -1 torque to the actuated joint | torque (N m) |
    | 1   | apply 0 torque to the actuated joint  | torque (N m) |
    | 2   | apply 1 torque to the actuated joint  | torque (N m) |

    ### Observation Space
    | Num | Observation                  | Min       | Max      |
    |-----|------------------------------|-----------|----------|
    | 0   | Cosine of `theta1`           | -1        | 1        |
    | 1   | Sine of `theta1`             | -1        | 1        |
    | 2   | Cosine of `theta2`           | -1        | 1        |
    | 3   | Sine of `theta2`             | -1        | 1        |
    | 4   | Angular velocity of `theta1` | (-4 * pi) | (4 * pi) |
    | 5   | Angular velocity of `theta2` | (-9 * pi) | (9 * pi) |
    where
    theta1 is the angle of the first joint, where an angle of 0 indicates the first link is pointing directly
    downwards. theta2 is relative to the angle of the first link
    The angular velocities of theta1 and theta2 are bounded at ±4pi, and ±9pi rad/s respectively.

    ### Rewards
    The goal is to have the free end reach a designated target height in as few steps as possible,
    and as such all steps that do not reach the goal incur a reward of -1.
    Achieving the target height results in termination with a reward of 0. The reward threshold is -100.

    ### Episode End
    The episode ends if the free end reaches the target height or
    the simulation length is greater than 500

    """

    try :
        from config import ACROBOT_CONFIG

        dT = ACROBOT_CONFIG['dT']
        LINK_LENGTH_1 = ACROBOT_CONFIG['LINK_LENGTH_1']
        LINK_LENGTH_2 = ACROBOT_CONFIG['LINK_LENGTH_2']
        LINK_MASS_1 = ACROBOT_CONFIG['LINK_MASS_1']
        LINK_MASS_2 = ACROBOT_CONFIG['LINK_MASS_2']
        LINK_COM_POS_1 = ACROBOT_CONFIG['LINK_COM_POS_1']
        LINK_COM_POS_2 = ACROBOT_CONFIG['LINK_COM_POS_2']
        LINK_MOI_1 = ACROBOT_CONFIG['LINK_MOI_1']
        LINK_MOI_2 = ACROBOT_CONFIG['LINK_MOI_2']
        MAX_VEL_1 = ACROBOT_CONFIG['MAX_VEL_1']
        MAX_VEL_2 = ACROBOT_CONFIG['MAX_VEL_2']
        AVAIL_TORQUE = ACROBOT_CONFIG['AVAIL_TORQUE']
        SCREEN_DIM = ACROBOT_CONFIG['SCREEN_DIM']
        MAX_STEPS = ACROBOT_CONFIG['MAX_STEPS']

    except ImportError as e:
        print(f"config file is missing or ACROBOT_CONFIG is missing from config file, error : {e}")
        sys.exit()
    except KeyError as e:
        print(f"ACROBOT_CONFIG is incorrect, error : {e}")
        sys.exit()

    

    def __init__(self):
        self.screen = pygame.display.set_mode((self.SCREEN_DIM, self.SCREEN_DIM))
        self._clock = pygame.time.Clock()
        self._state = None
        self._record, self._recorder = False, None
        self._render_flag = False
        self.done = False
        self._reward = 0
        self._steps = 0

        # height to which acrobot has to swing = total arm length/2 (default)
        self._BAR_HEIGHT = 0.5*(self.LINK_LENGTH_1 + self.LINK_LENGTH_2)

        bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 1  # 2.2 for default
        self.scale = self.SCREEN_DIM / (bound * 2)
        

    def reset(self, seed=None):
        if seed : np.random.seed(seed=seed)

        # random state between -0.1 rad to 0.1 rad
        # both links are pointing downwards when starting
        self._state = np.random.uniform(-0.1, 0.1, size=(4,)).astype(np.float32)
        self._record, self._recorder = False, None
        self._render_flag = False
        self.done = False
        self._reward = 0
        self._steps = 0

        return self._get_obs()

    def step(self, a):

        torque = self.AVAIL_TORQUE[a]

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(self._state, torque)
        ns = rk4(self._dsdt, s_augmented, [0, self.dT])
        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self._state = ns
        self._steps += 1
        if not self._render_flag : self._clock.tick() # increment the game loop only if render step is missing

        # checking if the state is terminal state
        self.done = bool(-self.LINK_LENGTH_1 * np.cos(self._state[0]) - self.LINK_LENGTH_2 * np.cos(self._state[1] + self._state[0]) > self._BAR_HEIGHT)
        self.reward = -1.0 if not self.done else 0.0
        
        # End the episode if stimulations cross 500
        if self._steps > self.MAX_STEPS :
            self.done, self.reward = True, 0.0

        return (self._get_obs(), self.reward, self.done)

    def _get_obs(self):
        return np.array(
            [np.cos(self._state[0]), np.sin(self._state[0]), np.cos(self._state[1]), np.sin(self._state[1]), self._state[2], self._state[3]], dtype=np.float32)

    def _dsdt(self, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI_1
        I2 = self.LINK_MOI_2
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = (
            m1 * lc1**2
            + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * np.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2)
            + phi2
        )
        ddtheta2 = (
            a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * np.sin(theta2) - phi2
        ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

    def render(self, FPS_lock:int=None, debug=False):

        # setting up the render flag showing that this
        # will handle the update rate
        self._render_flag = True
        surf = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surf.fill((211,211,211))
        s = self._state

        offset = self.SCREEN_DIM / 2

        p1 = [
            -self.LINK_LENGTH_1 * np.cos(s[0]) * self.scale,
            self.LINK_LENGTH_1 * np.sin(s[0]) * self.scale,
        ]

        p2 = [
            p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]) * self.scale,
            p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1]) * self.scale,
        ]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - np.pi / 2, s[0] + s[1] - np.pi / 2]
        link_lengths = [self.LINK_LENGTH_1 * self.scale, self.LINK_LENGTH_2 * self.scale]

        pygame.draw.aaline(
            surf, color=(0, 0, 0),
            start_pos=(-2 * self.scale + offset, self._BAR_HEIGHT * self.scale + offset),
            end_pos=(2 * self.scale + offset, self._BAR_HEIGHT * self.scale + offset),
            blend=True
        )

        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1 * self.scale, -0.1 * self.scale
            coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)

            pygame.draw.aalines(surf, (0,0,0), True,transformed_coords)

            pygame.draw.circle(surf, (113, 125, 126), (int(x), int(y)), int(0.1 * self.scale))
            pygame.draw.circle(surf, (0,0,0), (int(x), int(y)), int(0.1 * self.scale),1)
            
        if debug : 
            mid_point = ((transformed_coords[-1][0]+transformed_coords[-2][0])/2, # mid point for 2nd link edge
                         (transformed_coords[-1][1]+transformed_coords[-2][1])/2)
            pygame.draw.aaline(surf,(241, 38, 11),
                                start_pos=mid_point,
                                end_pos=(mid_point[0],self._BAR_HEIGHT * self.scale + offset),blend=True)

        surf = pygame.transform.flip(surf, False, True)
        if debug : surf = self.__render_stats(surf)
        self.screen.blit(surf, (0, 0))

        pygame.event.pump()
        if FPS_lock : self._clock.tick(FPS_lock)
        else : self._clock.tick()
        pygame.display.flip()
        
    def __render_stats(self,surf):
        '''
        rendering the acrobot state and other stats
        '''

        color = (241, 38, 11)
        text_surface = Acrobot.FONT.render(f"Map size : {self.SCREEN_DIM/self.scale :.2f} m x {self.SCREEN_DIM/self.scale :.2f} m",True,color)
        surf.blit(text_surface,(10,15))
        text_surface = Acrobot.FONT.render(f"Theta 1 : {int(self._state[0]*180/np.pi)} deg",True,color)
        surf.blit(text_surface,(10,30))
        text_surface = Acrobot.FONT.render(f"Theta 2 : {int(self._state[1]*180/np.pi)} deg",True,color)
        surf.blit(text_surface,(10,45))
        text_surface = Acrobot.FONT.render(f"Omega 1 : {int(self._state[2]*180/np.pi)} deg/s",True,color)
        surf.blit(text_surface,(10,60))
        text_surface = Acrobot.FONT.render(f"Omega 2 : {int(self._state[3]*180/np.pi)} deg/s",True,color)
        surf.blit(text_surface,(10,75))

        text_surface = Acrobot.FONT.render(f"FPS : {int(self._clock.get_fps())}",True,color)
        surf.blit(text_surface,(self.SCREEN_DIM-110,15))
        text_surface = Acrobot.FONT.render(f"Reward : {self.reward}",True,color)
        surf.blit(text_surface,(self.SCREEN_DIM-110,30))
        text_surface = Acrobot.FONT.render(f"Steps : {self._steps}",True,color)
        surf.blit(text_surface,(self.SCREEN_DIM-110,45))
        text_surface = Acrobot.FONT.render(f"done : {self.done}",True,color)
        surf.blit(text_surface,(self.SCREEN_DIM-110,60))

        return surf

    def close_quit(self):
        '''
        To avoid crash press the exit button or press 'q'
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_q):
                self.done = True
                if self._record : self._recorder.release()
                pygame.display.quit()
                pygame.quit()
    
    def record(self,filename,FPS:int=15):
        '''
        filename without extension

        reference : https://github.com/tdrmk/pygame_recorder
        '''
        if not self._render_flag :
            print("Render the scene to recorder")
            sys.exit()
        if self._record==False and self.done==False :
            self._recorder = cv2.VideoWriter(f'{filename}.mp4',0x7634706d,float(FPS),(self.SCREEN_DIM,self.SCREEN_DIM))
            print(f'Environment recording will be saved to {filename}.mp4')
            self._record = True
        
        pixels = cv2.rotate(pygame.surfarray.pixels3d(self.screen), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        self._recorder.write(pixels)

        if self._record==True and self.done==True :
            self._recorder.release()


def wrap(x, m, M):
    '''
    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range
    Returns:
        x: a scalar, wrapped
    '''
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    '''
    Args:
        x: scalar
        m: The lower bound
        M: The upper bound
    Returns:
        x: scalar, bound between min (m) and Max (M)
    '''
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t):
    '''
    Integrate 1-D or N-D system of ODEs using 4-th order Runge-Kutta.
    Args:
        derivs: the derivative of the system and has the signature ``dy = derivs(yi)``
        y0: initial state vector
        t: sample times
    Returns:
        yout: Runge-Kutta approximation of the ODE
    '''

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        this = t[i]
        dT = t[i + 1] - this
        dt2 = dT / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0))
        k2 = np.asarray(derivs(y0 + dt2 * k1))
        k3 = np.asarray(derivs(y0 + dt2 * k2))
        k4 = np.asarray(derivs(y0 + dT * k3))
        yout[i + 1] = y0 + dT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    # We only care about the final timestep and we cleave off action value which will be zero
    return yout[-1][:4]


def main():
    # Creating a Environment instance
    env = Acrobot()

    observation = env.reset(seed=42)
    eps_reward = 0
    while not env.done:
        action = np.random.randint(0,3)
        observation, reward, done = env.step(action)
        eps_reward += reward
        env.render(FPS_lock=15,debug=True)

        # Example to fix the update rate and rendering rate to 30 FPS,show car stats
        # env.render_env(FPS_lock=30,render_stats=True)

        # Example to decouple the update rate from rendering rate,show car stats
        # env.render_env(FPS_lock=None,render_stats=True)

        # Example to record the showing Environment
        # env.record("output",FPS=15)

        env.close_quit()
    

if __name__ == "__main__":
    main()
