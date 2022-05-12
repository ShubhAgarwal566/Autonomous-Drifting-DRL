import sys
import gym
import time
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from argparse import Namespace
import argparse
import yaml
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch',  default=8800, type=int)
parser.add_argument('--vel_max',  default=6.0, type=float)
args = parser.parse_args()

throttle_range = [0.4, 1.0]
steer_range = [-0.8, 0.8]

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim ,min_log_std=-20, max_log_std=2):##max and min left to modify
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(state_dim, 512)
		self.fc2 = nn.Linear(512, 256)
		self.mu_head = nn.Linear(256, action_dim)
		self.log_std_head = nn.Linear(256, action_dim)
		self.min_log_std = min_log_std
		self.max_log_std = max_log_std

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		mu = self.mu_head(x)
		log_std_head = self.log_std_head(x)
		#clamp same as clip
		log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std) ##give a resitriction on the chosen action
		return mu, log_std_head

	def test(self, state):
		state = torch.FloatTensor(state).to(device)
		mu, log_sigma = self(state)
		action = mu
		steer = float(torch.tanh(action[0,0]).detach().cpu().numpy())
		throttle = float(torch.tanh(action[0,1]).detach().cpu().numpy())

		steer = (steer + 1)/2 * (steer_range[1] - steer_range[0]) + steer_range[0]
		throttle = (throttle + 1)/2 * (throttle_range[1] - throttle_range[0]) + throttle_range[0]

		return np.array([[steer, throttle]])



if __name__ == "__main__":

	with open('config_example_map.yaml') as file:
		conf_dict = yaml.load(file, Loader=yaml.FullLoader)
	conf = Namespace(**conf_dict)

	#init params
	params_dict = {'mu': 0.5, #1.0489,
					'C_Sf': 4.718,
					'C_Sr': 5.4562,
					'lf': 0.15875,
					'lr': 0.17145,
					'h': 0.074,
					'm': 3.74,
					'I': 0.04712,
					's_min': -0.4189,
					's_max': 0.4189,
					'sv_min': -3.2,
					'sv_max': 3.2,
					'v_switch':7.319,
					'a_max': 9.51,
					'v_min':-5.0,
					'v_max': 20.0,
					'width': 0.31,
					'length': 0.58}

	env = gym.make('f110_gym:f110-v0',params=params_dict,  map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
	obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
	env.render(mode='human_fast')

	action_dim = 2 # steer, throttle
	state_dim = 42 
	print('action_dimension:', action_dim, ' --- state_dimension:', state_dim)
	
	# Initializing the Agent for SAC and load the trained weights
	actor = Actor(state_dim=state_dim, action_dim = action_dim).to(device)
	model_path = 'SAC_model_50000/policy_net_' + str(args.epoch) + '.pth'
	actor.load_state_dict(torch.load( model_path ))

	destinationFlag = False
	
	t0 = time.time()
	first_step_pass = False

	#give little throttle at start
	env.step(np.array([[0,0.5*args.vel_max]]))

	while(True):
		env.render(mode='human_fast')

		if not first_step_pass:
			steer = 0.0
			throttle = 0.0
		else:
			action = actor.test(tState)
			action = np.reshape(action, [1,2])

			steer = action[0,0]
			throttle = action[0,1]	

		next_state, reward, destinationFlag, _ = env.step(np.array([[steer, throttle*args.vel_max]]))
		next_state = np.reshape(next_state, [1, state_dim])
		
		tState = next_state
		endFlag = destinationFlag

		if endFlag:
			break
				
		first_step_pass = True
		