# from train_policy import train_discrete, test_discrete
import train_policy
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
# from driving_policy import DiscreteDrivingPolicy
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset, DataLoader
# from dataset_loader import DrivingDataset


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
    import logging
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(filename='dagger.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logger = logging.getLogger("dagger")
    logger.info('Test')
    
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
    parser.add_argument("--resume_iter", type=int, help="the iteration of resumed model", default=0)
    parser.add_argument("--resume_path", type=str, help="the path of resumed model", default='')
    parser.add_argument("--local_expert", type=bool, help="run expert demonstration locally", default=False)
    args = parser.parse_args()
    args.save_expert_actions = True
    # args.out_dir = args.train_dir
    args.expert_drives = False

    #####
    ## Enter your DAgger code here
    ## Reuse functions in racer.py and train_policy.py
    ## Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where 
    #####

    ## local expert demonstration
    if args.local_expert == True:
        policy = torch.load(args.resume_path, map_location=torch.device('cpu')).to(DEVICE)
        args.run_id = args.resume_iter
        args.out_dir = f'./dataset/{iter}/'
        print('GETTING EXPERT DEMONSTRATIONS')
        error_heading, error_dist, dest_min = run(policy, args)
        logger.info(f"Result at iter {args.run_id - 1}: {error_heading}, {error_dist}, {dest_min}")

    else:
        args.weights_out_file = f'./weights/learn_{args.resume_iter}.weights'
        print('RETRAINING LEARNER ON AGGREGATED DATASET')
        print(f'ITERATION: {args.resume_iter}')
        logger.info(f"Retraining at iteration {args.resume_iter}")
        policy = train_policy.main(args)


        
    



    