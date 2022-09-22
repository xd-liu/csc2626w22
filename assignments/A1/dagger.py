import train_policy
import racer
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
import os

from utils import DEVICE, str2bool
from full_state_car_racing_env import FullStateCarRacingEnv
import imageio


def run(steering_network, args):
    
    env = FullStateCarRacingEnv()
    env.reset()
    
    learner_action = np.array([0.0, 0.0, 0.0])
    expert_action = None
    
    for t in range(args.timesteps):
        env.render()
        
        state, expert_action, reward, done, _ = env.step(learner_action) 
        if done:
            break
        
        expert_steer = expert_action[0]  # [-1, 1]
        expert_gas = expert_action[1]    # [0, 1]
        expert_brake = expert_action[2]  # [0, 1]

        if args.expert_drives:
            learner_action[0] = expert_steer
        else:
            learner_action[0] = steering_network.eval(state, device=DEVICE)
            
        learner_action[1] = expert_gas
        learner_action[2] = expert_brake

        if args.save_expert_actions:
            imageio.imwrite(os.path.join(args.out_dir, 'expert_%d_%d_%f.jpg' % (args.run_id, t, expert_steer)), state)

    error_heading, error_dist, dest_min = env.get_cross_track_error(env.car, env.track)
    env.close()
    
    return error_heading, error_dist, dest_min


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--dagger_iterations", type=int, help="", default=10)

    ## new added arguments
    parser.add_argument("--timesteps", type=int, help="timesteps of simulation to run to get aggregated data", default=100000)
    args = parser.parse_args()
    args.save_expert_actions = True
    args.out_dir = args.train_dir
    args.expert_drives = False

    #####
    ## Enter your DAgger code here
    ## Reuse functions in racer.py and train_policy.py
    ## Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where 
    #####
    args.weights_out_file = 'weights/learn_0.weights'
    print('TRAINING LEARNER ON INITIAL DATASET')
    policy = train_policy.main(args)
    
    error_heading_list = []
    error_dist_list = []
    dest_min_list = []

    for iter in range(1, args.dagger_iterations + 1):
        args.run_id = iter

        print('GETTING EXPERT DEMONSTRATIONS')
        error_heading, error_dist, dest_min = run(policy, args)
        print('error_heading', error_heading)
        error_heading_list.append(error_heading)

        print('error_dist', error_dist)
        error_dist_list.append(error_dist)

        print('dest_min', dest_min)
        dest_min_list.append(dest_min)

        print('RETRAINING LEARNER ON AGGREGATED DATASET')
        print(f'ITERATION: {iter}')
        args.weights_out_file = f'learn_{iter}.weights'
        policy = train_policy.main(args)
    
    # final
    error_heading, error_dist, dest_min = run(policy, args)
    print('error_heading', error_heading)
    error_heading_list.append(error_heading)

    print('error_dist', error_dist)
    error_dist_list.append(error_dist)

    print('dest_min', dest_min)
    dest_min_list.append(dest_min)

    print('error_heading_list', error_heading_list)
    print('error_dist_list', error_dist_list)
    print('dest_min_list', dest_min_list)

    error_heading_list = np.array(error_heading_list)
    np.save('error_heading_list.npy', error_heading_list)

    error_dist_list = np.array(error_dist_list)
    np.save('error_dist_list.npy', error_dist_list)

    dest_min_list = np.array(dest_min_list)
    np.save('dest_min_list.npy', dest_min_list)



        
    



    