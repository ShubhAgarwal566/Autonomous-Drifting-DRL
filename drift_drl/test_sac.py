import sys
from environment import *
import time
import random
import os
import torch
import gym

np.random.seed(1234)
device = 'cuda' if torch.cuda.is_available() else 'cpu'



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

		return np.array([steer, throttle])



if __name__ == "__main__":

	pygame.init()
	pygame.font.init()


	#init environment
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
    work['tlad'] =  3
    # work['vgain'] = 0.25
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
    env.render()

	###################

	
	action_dim = 2 # steer, throttle
	# state = env.getState()
	# state_dim = len(state)
	state_dim = 42 
	print('action_dimension:', action_dim, ' --- state_dimension:', state_dim)
	
	# Initializing the Agent for SAC and load the trained weights
	actor = Actor(state_dim=state_dim, action_dim = action_dim).to(device)
	model_path = '../weights/sac.pth'
	actor.load_state_dict(torch.load( model_path ))

	destinationFlag = False
	
	# os.makedirs('./test/' + model ,exist_ok=True)
	# for setup in setups:
	print('TESTING: setup: ',setup)
	# save_path = './test/' + model +'/' +str(setup[0])+'_'+str(setup[1])
	# save_file = open(save_path + '.csv','w')
	# writer = csv.writer(save_file)
	# writer.writerow(headers)
	# env.reset(traj_num=6, testFlag=True, test_friction=setup[0], test_mass=setup[1])

	t0 = time.time()
	first_step_pass = False

	#give little throttle at start
	env.step(np.array([0,0.5]))

	while(True):
		env.render()

		# if time.time()-t0 < 0.5:
		# 	# make sure the collision sensor is empty
		# 	env.world.collision_sensor.history = []

		if time.time()-t0 > 0.5:

			if not first_step_pass:
				steer = 0.0
				throttle = 0.0
				# hand_brake = False
			else:
				action = actor.test(tState)
				action = np.reshape(action, [1,2])

				steer = action[0,0]
				throttle = action[0,1]	

			next_state, reward, destinationFlag, _ = env.step(steer=steer, throttle=throttle)
			next_state = np.reshape(next_state, [1, state_dim])
			
			tState = next_state

			######################################################3
			# # prepare the state information to be saved
	
			# t = time.time() - t0
			# location = env.world.player.get_location()
			# wx = location.x
			# wy = location.y
			# course = getHeading(env)
			# vx = env.velocity_local[0]
			# vy = env.velocity_local[1]
			# speed = np.sqrt(vx*vx + vy*vy)
			# slip_angle = env.velocity_local[2]
			# cte = tState[0,2]
			# cae = tState[0,4]
			# traj_index = env.traj_index
			# steer = control.steer
			# throttle = control.throttle
			# cf = bool2num(collisionFlag)
			# df = bool2num(destinationFlag)
			# af = bool2num(awayFlag)
			
			# # save to the csv file for further analysis
			# print('time stamp: ', t)
			# writer.writerow([t,wx,wy,course,vx,vy,speed,slip_angle,cte,cae,traj_index,reward,steer,throttle,cf,df,af])
			
			###################################################

			endFlag = destinationFlag

			if endFlag:
				break
				
			
			first_step_pass = True
		