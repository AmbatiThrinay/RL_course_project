import numpy as np

ACROBOT_CONFIG = {

    'dT' : 0.2,

    'LINK_LENGTH_1' : 1.0, # [m]
    'LINK_LENGTH_2' : 1.0, # [m]
    'LINK_MASS_1' : 1.0, # [kg] mass of link 1
    'LINK_MASS_2' : 1.0, # [kg] mass of link 2
    'LINK_COM_POS_1' : 1.0/2, # LINK_LENGTH_1/2 in [m] position of the center of mass of link 1
    'LINK_COM_POS_2' : 1.0/2, # LINK_LENGTH_2/2 in [m] position of the center of mass of link 2
    'LINK_MOI_1' : 1.0, # moments of inertia for link 1
    'LINK_MOI_2' : 1.0, # moments of inertia for link 1

    'MAX_VEL_1' : 4 * np.pi, # [rad/sec] maximum angular velocity of link 1
    'MAX_VEL_2' : 9 * np.pi, # [rad/sec] maximum angular velocity of link 2
    'AVAIL_TORQUE' : [-1.0, 0.0, +1], # Available actions

    'SCREEN_DIM' : 700, # pygame screen dimensions
    'MAX_STEPS' : 600 # number simulation steps after which the episode ends
}

DQN_CONFIG = {

    "nn_lr" : 1e-3, # Learning rate for neural network
    "discount" : 0.99, # discount factor
    "batch_size" : 32, # minibatch size for training neural network
    "seed" : 333, # seed for random generator for intializing network weight
    "eps_max" : 0.95, # eplison starting value
    "eps_min" : 0.01, # final annealed exploration rate
    "eps_decay_rate" : 1e-3, # decay rate of the epsilon
    # n_episode for eps to reach minimum = (eps_max-eps_min)/(eps_decay_rate*episodes)
    "memory_size" : 100_000, # size of the replay buffer
    "target_update_rate" : 100, # update rate of target network

}
