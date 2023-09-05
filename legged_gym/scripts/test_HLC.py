import numpy as np
import os
from datetime import datetime
from isaacgym import gymutil

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def run_LLC(args):
    """
    Run the low level policy
    """
    # parsing arguments
    sim_device_type, sim_device_id = gymutil.parse_device_str(args.sim_device)
    if sim_device_type=='cuda' and args.use_gpu_pipeline:
        device = args.sim_device
    else:
        device = 'cpu'

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5 # TODO: Figure out what num_rows and num_cols do
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.env.HLC_test_sim = True # WASD control and custom map

    des_command = torch.zeros(env_cfg.env.num_envs, 3, dtype=torch.float, device=device, requires_grad=False)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # set-up keyboard control
    def turn_right():
        des_command[:, 2] = -0.05

    def turn_left():
        des_command[:, 2] = 0.05
    
    def move_forward():
        des_command[:, 0] = 0.3

    def stop():
        des_command[:, 0] = 0.0
        des_command[:, 1] = 0.0
        des_command[:, 2] = 0.0
    
    env.set_robot_keyboard_control_funcs(turn_right, turn_left, move_forward, stop)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach(), des_command)


if __name__ == '__main__':
    args = get_args()
    run_LLC(args)
