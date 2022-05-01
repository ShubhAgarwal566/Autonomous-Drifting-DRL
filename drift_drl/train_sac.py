import sys
import gym
from SACAgent import *
import time
import random
from argparse import Namespace
import yaml

########SAC#######
if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
	parser.add_argument('--target_update_interval', default=1, type=int)
	parser.add_argument('--gradient_steps', default=1, type=int)
	parser.add_argument('--vel_max', default=6.0, type=int)

	parser.add_argument('--learning_rate', default=3e-4, type=int)
	parser.add_argument('--gamma', default=0.99, type=int) # discount gamma

	parser.add_argument('--capacity', default=50000, type=int) # replay buffer size
	parser.add_argument('--iteration', default=100000, type=int) #  num of  games
	parser.add_argument('--batch_size', default=512, type=int) # mini batch size


	parser.add_argument('--seed', default=1, type=int)

	# optional parameters
	parser.add_argument('--num_hidden_layers', default=2, type=int)
	parser.add_argument('--num_hidden_units_per_layer', default=256, type=int)
	parser.add_argument('--sample_frequency', default=256, type=int)
	parser.add_argument('--activation', default='Relu', type=str)
	parser.add_argument('--render', default=False, type=bool) # show UI or not
	parser.add_argument('--log_interval', default=50, type=int) #
	parser.add_argument('--load', default=False, type=bool) # load model

	args = parser.parse_args()

	#code taken from waypoint_follow.py
	#tlad is lookahead distance

	work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
	work['tlad'] =  3.0
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

	#########################
	
	action_dim = 2
	state_dim = 42
	print('action_dimension:', action_dim, ' & state_dimension:', state_dim)

	destinationFlag = False
	collisionFlag = False
	f110_startFlag = False

	agent = SACAgent(state_dim=state_dim, action_dim=action_dim)

	print("====================================")
	print("Collection Experience...")
	print("====================================")

	ep_r = 0###expectation of reward R
	for i in range(args.iteration):
		# TODO: change this for f110
		_ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

		t0 = time.time()
		#check later
		first_step_pass = False

		count = 0
		speed = 0
		cte = 0 #cross track error
		hae = 0 #heading angle error
		time_cost = 0

		while(True): # one episode
			count += 1
			env.render()

			#give little throttle at start
			if f110_startFlag==False:
				env.step(np.array([[0,0.5*args.vel_max]]))
				f110_startFlag = True
				continue

			# if time.time()-t0 < 0.5: #ignore any collisions that happen in the first 0.5 seconds
				# env.world.collision_sensor.history = []

			#except n*10th until 3000 
			if i % 5 != 0 or agent.replay_buffer.num_transition <= 3000:
				#reduce the refresh rate of the control loop
				if time.time()-t0 > 0.5:
					if not first_step_pass:
						steer = 0.0
						throttle = 0.0
						# hand_brake = False
					else:
						action = agent.select_action(tState)
						# print(action.shape)
						action = np.reshape(action, [1,2])
						# print(action.shape)

						steer = action[0,0]
						throttle = action[0,1]
						print("mapped steer: ", steer, ", throttle: ",throttle)
						if i%5==0:
							agent.writer.add_scalar('Control/iteration_'+str(i)+'/steer', steer, global_step = count)
							agent.writer.add_scalar('Control/iteration_'+str(i)+'/throttle', throttle, global_step = count) 

					# TODO: change this f110
					next_state, reward, destinationFlag, _= env.step(np.array([[steer, throttle*args.vel_max]]))
					
					# next_state, reward, collisionFlag, destinationFlag, awayFlag, control = env.step(steer, throttle)
					next_state = np.reshape(next_state, [1, state_dim])
					
					ep_r += reward
					
					if first_step_pass: 
						
						action[0,0] = (action[0,0] - agent.steer_range[0]) / (agent.steer_range[1] - agent.steer_range[0]) * 2 - 1
						action[0,1] = (action[0,1] - agent.throttle_range[0]) / (agent.throttle_range[1] - agent.throttle_range[0]) * 2 - 1

						# TODO: look for changes for f110
						agent.replay_buffer.push(tState, action, reward, next_state, destinationFlag)
					
					tState = next_state
					
					vx = env.vel_x
					vy = env.vel_y
					speed += np.sqrt(vx*vx + vy*vy)
					# cte = cross track error, check https://medium.com/asap-report/introduction-to-the-carla-simulator-training-a-neural-network-to-control-a-car-part-1-e1c2c9a056a5
					# Use variable e_y or e_dist
					cte += tState[0,2]
					# heading error angle, check notes on tablet. Use variable e_psi
					hae += abs(tState[0,4])

					if destinationFlag:
						break
					
					print('buffer_size: %d'%agent.replay_buffer.num_transition)
					
					first_step_pass = True 
			else: #goes here every 10th iteration and every iter after 3000  ## TESTING BLOCK ##
				#reduce the refresh rate of the control loop
				if time.time()-t0 > 0.5:

					#TODO (ROCKET): Remove this and test after a succesful run
					if not first_step_pass:
						steer = 0.0
						throttle = 0.0
						# hand_brake = False
					else:
						action = agent.test(tState)
						action = np.reshape(action, [1,2])

						steer = action[0,0]
						throttle = action[0,1]
						print('############### TESTING ##############')
						print("mapped steer: ", steer, ", throttle: ",throttle)
						if i%5==0:
							agent.writer.add_scalar('TEST/Control/iteration_'+str(i)+'/steer', steer, global_step = count)
							agent.writer.add_scalar('TEST/Control/iteration_'+str(i)+'/throttle', throttle, global_step = count)    

					next_state, reward, destinationFlag, _ = env.step(np.array([[steer, throttle*args.vel_max]]))
					next_state = np.reshape(next_state, [1, state_dim])
					ep_r += reward
					
					tState = next_state
									
					vx = env.vel_x
					vy = env.vel_y
					speed += np.sqrt(vx*vx + vy*vy)
					cte += tState[0,2]
					hae += abs(tState[0,4])

					if destinationFlag:
						break

					first_step_pass = True 
		
		time_cost = time.time() - t0

		#starts with ignoring bad data and then gradually increase the number of training cycles        
		if i % 10 != 0 or agent.replay_buffer.num_transition <= 3000:
			print("*************TRAIN**************")
			if agent.replay_buffer.num_transition >= 1000 and agent.replay_buffer.num_transition<10000:
				for u in range(5):
					agent.update()
			if agent.replay_buffer.num_transition >= 10000 and agent.replay_buffer.num_transition<40000:
				for u in range(100):
					agent.update()
			if agent.replay_buffer.num_transition>=40000 and agent.replay_buffer.num_transition<80000:
				for u in range(300):
					agent.update()
			if agent.replay_buffer.num_transition>=80000 and agent.replay_buffer.num_transition<150000:
				for u in range(400):
					agent.update()
			if agent.replay_buffer.num_transition>=150000 and agent.replay_buffer.num_transition<300000:
				for u in range(600):
					agent.update()
			if agent.replay_buffer.num_transition>=300000:
				for u in range(800):
					agent.update()
			print("***********TRAIN OVER***********")

		
		speed = speed / count
		cte = cte/count
		hae = hae/count

		if i % 10 == 0 and agent.replay_buffer.num_transition > 3000:
			agent.save(i, args.capacity)
		
		print("Ep_i: %d, the ep_r is: %.2f" % (i, ep_r))

		agent.writer.add_scalar('Metrics/ep_r', ep_r, global_step=i)
		agent.writer.add_scalar('Metrics/time_cost', time_cost, global_step=i)
		agent.writer.add_scalar('Metrics/avg_speed', speed, global_step=i)
		agent.writer.add_scalar('Metrics/avg_cross_track_error', cte, global_step=i)
		agent.writer.add_scalar('Metrics/avg_heading_error', hae, global_step=i)
		agent.writer.add_scalar('Metrics/reward_every_second', ep_r/time_cost, global_step=i)

		# agent.writer.add_scalar('Physics/Tire_friction', env.tire_friction, global_step = i)
		# agent.writer.add_scalar('Physics/Mass', env.mass, global_step=i)
		

		if i % 10 ==0 and agent.replay_buffer.num_transition > 3000:
			agent.writer.add_scalar('Metrics_test/ep_r', ep_r, global_step=i)
			agent.writer.add_scalar('Metrics_test/time_cost', time_cost, global_step=i)
			agent.writer.add_scalar('Metrics_test/avg_speed', speed, global_step=i)
			agent.writer.add_scalar('Metrics_test/avg_cross_track_error', cte, global_step=i)
			agent.writer.add_scalar('Metrics_test/avg_heading_error', hae, global_step=i)
			agent.writer.add_scalar('Metrics_test/reward_every_second', ep_r/time_cost, global_step=i)

			# agent.writer.add_scalar('Physics_test/Tire_friction', env.tire_friction, global_step = i)
			# agent.writer.add_scalar('Physics_test/Mass', env.mass, global_step=i)
		ep_r = 0
