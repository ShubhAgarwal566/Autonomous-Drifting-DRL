import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt
from numba import njit
import pandas as pd
from numpy import sqrt, pi

from pyglet.gl import GL_POINTS

"""
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
	"""
	Return the nearest point along the given piecewise linear trajectory.

	Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
	not be an issue so long as trajectories are not insanely long.

		Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

	point: size 2 numpy array
	trajectory: Nx2 matrix of print(x,y) trajectory waypoints
		- these must be unique. If they are not unique, a divide by 0 error will destroy the world
	"""
	diffs = trajectory[1:,:] - trajectory[:-1,:]
	l2s   = diffs[:,0]**2 + diffs[:,1]**2
	# this is equivalent to the elementwise dot product
	# dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
	dots = np.empty((trajectory.shape[0]-1, ))
	for i in range(dots.shape[0]):
		dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
	t = dots / l2s
	t[t<0.0] = 0.0
	t[t>1.0] = 1.0
	# t = np.clip(dots / l2s, 0.0, 1.0)
	projections = trajectory[:-1,:] + (t*diffs.T).T
	# dists = np.linalg.norm(point - projections, axis=1)
	dists = np.empty((projections.shape[0],))
	for i in range(dists.shape[0]):
		temp = point - projections[i]
		dists[i] = np.sqrt(np.sum(temp*temp))
	min_dist_segment = np.argmin(dists)
	return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
	"""
	starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

	Assumes that the first segment passes within a single radius of the point

	http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
	"""
	start_i = int(t)
	start_t = t % 1.0
	first_t = None
	first_i = None
	first_p = None
	trajectory = np.ascontiguousarray(trajectory)
	for i in range(start_i, trajectory.shape[0]-1):
		start = trajectory[i,:]
		end = trajectory[i+1,:]+1e-6
		V = np.ascontiguousarray(end - start)

		a = np.dot(V,V)
		b = 2.0*np.dot(V, start - point)
		c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
		discriminant = b*b-4*a*c

		if discriminant < 0:
			continue
		#   print "NO INTERSECTION"
		# else:
		# if discriminant >= 0.0:
		discriminant = np.sqrt(discriminant)
		t1 = (-b - discriminant) / (2.0*a)
		t2 = (-b + discriminant) / (2.0*a)
		if i == start_i:
			if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
				first_t = t1
				first_i = i
				first_p = start + t1 * V
				break
			if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
				first_t = t2
				first_i = i
				first_p = start + t2 * V
				break
		elif t1 >= 0.0 and t1 <= 1.0:
			first_t = t1
			first_i = i
			first_p = start + t1 * V
			break
		elif t2 >= 0.0 and t2 <= 1.0:
			first_t = t2
			first_i = i
			first_p = start + t2 * V
			break
	# wrap around to the beginning of the trajectory if no intersection is found1
	if wrap and first_p is None:
		for i in range(-1, start_i):
			start = trajectory[i % trajectory.shape[0],:]
			end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
			V = end - start

			a = np.dot(V,V)
			b = 2.0*np.dot(V, start - point)
			c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
			discriminant = b*b-4*a*c

			if discriminant < 0:
				continue
			discriminant = np.sqrt(discriminant)
			t1 = (-b - discriminant) / (2.0*a)
			t2 = (-b + discriminant) / (2.0*a)
			if t1 >= 0.0 and t1 <= 1.0:
				first_t = t1
				first_i = i
				first_p = start + t1 * V
				break
			elif t2 >= 0.0 and t2 <= 1.0:
				first_t = t2
				first_i = i
				first_p = start + t2 * V
				break

	return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
	"""
	Returns actuation
	"""
	waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
	speed = lookahead_point[2]
	if np.abs(waypoint_y) < 1e-6:
		return speed, 0.
	radius = 1/(2.0*waypoint_y/lookahead_distance**2)
	steering_angle = np.arctan(wheelbase/radius)
	return speed, steering_angle

class PurePursuitPlanner:
	"""
	Example Planner
	"""
	def __init__(self, conf, wb):
		self.wheelbase = wb
		self.conf = conf
		self.load_waypoints(conf)
		self.max_reacquire = 20.
		# print(conf.wpt_path)
		self.drawn_waypoints = []

	def load_waypoints(self, conf):
		"""
		loads waypoints
		"""
		self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
		# print(self.waypoints.shape)
	def render_waypoints(self, e):
		"""
		update waypoints being drawn by EnvRenderer
		"""

		#points = self.waypoints

		points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
		
		scaled_points = 50.*points
		# print(scaled_points)
		for i in range(points.shape[0]):
			if len(self.drawn_waypoints) < points.shape[0]:
				b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
								('c3B/stream', [183, 193, 222]))
				self.drawn_waypoints.append(b)
			else:
				self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
		
	def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
		"""
		gets the current waypoint to follow
		"""
		wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
		nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
		if nearest_dist < lookahead_distance:
			lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
			if i2 == None:
				return None
			current_waypoint = np.empty((3, ))
			# x, y
			current_waypoint[0:2] = wpts[i2, :]
			# speed
			current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
			return current_waypoint
		elif nearest_dist < self.max_reacquire:
			return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
		else:
			return None

	def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
		"""
		gives actuation given observation
		"""
		position = np.array([pose_x, pose_y])
		lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

		if lookahead_point is None:
			return 4.0, 0.0

		speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
		speed = vgain * speed

		return speed, steering_angle

def main(route):
	"""
	main entry point
	"""

	work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.90338203837889}
	work['tlad'] =  3
	# work['vgain'] = 0.25
	with open('config_example_map.yaml') as file:
		conf_dict = yaml.load(file, Loader=yaml.FullLoader)
	conf = Namespace(**conf_dict)
	# print(conf_dict)
	planner = PurePursuitPlanner(conf, 0.17145+0.15875)

	def render_callback(env_renderer):
		# custom extra drawing function

		e = env_renderer

		# update camera to follow car
		x = e.cars[0].vertices[::2]
		y = e.cars[0].vertices[1::2]
		top, bottom, left, right = max(y), min(y), min(x), max(x)
		e.score_label.x = left
		e.score_label.y = top - 700
		e.left = left - 800
		e.right = right + 800
		e.top = top + 800
		e.bottom = bottom - 800

		planner.render_waypoints(env_renderer)


	#init params
	params_dict = {'mu': 1.0489,
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
	env.add_render_callback(render_callback)
	
	obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
	env.render()

	laptime = 0.0
	start = time.time()
	slip_angles_list = []
	e_heading_list = []
	e_slip_list = []
	while not done:
		slip_angles_list.append(obs['slip_angles'][0])
		reward = get_reward(route, obs, e_heading_list, e_slip_list)
		speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
		speed = max(speed, 4.5)
		obs, step_reward, done, info = env.step(np.array([[steer, speed]]), render=True)
		laptime += step_reward
		env.render(mode='human')
		# done = False
	
	print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)
	
	print("Min of error heading: %s"%min(e_heading_list))
	print("Max of error heading: %s"%max(e_heading_list))

	print("\nMin of error slip: %s"%min(e_slip_list))
	print("Max of error slip: %s"%max(e_slip_list))

	plt.figure()
	plt.plot(slip_angles_list)
	plt.title("Slip Angle vs Time")
	plt.xlabel("Time")
	plt.ylabel("Slip Angle")
	plt.savefig('waypoints_slip.png')
	print("Plot saved successfully")

def get_closest(route, current_x, current_y):
	dists = sqrt( (route[:,0]-current_x)**2 + (route[:,1]-current_y)**2 ) # distance to all points
	closest_idx	= np.argmin(dists.reshape(-1)) # idx of point with least distance
	return closest_idx

def get_reward(route, obs, e_heading_list, e_slip_list):
	closest_idx = get_closest(route, obs['poses_x'][0], obs['poses_y'][0])
	ref_point  = route[closest_idx].reshape(-1)

	e_dis = abs(ref_point[1] - obs['poses_y'][0])
	e_slip = ref_point[4] - obs['slip_angles'][0] 
	e_heading = ref_point[2] - obs['poses_theta'][0]
	if(e_heading>pi):
		e_heading -= 2*pi
	if(e_heading<-pi):
		e_heading += 2*pi


	r_dis = np.exp(-0.5*e_dis)

	if abs(e_heading)<pi/2:
		r_heading = np.exp(-0.1*abs(e_heading))
	elif (e_heading)>= pi/2:
		r_heading = -np.exp(-0.1*(pi-e_heading))
	else:
		r_heading = -np.exp(-0.1*(e_heading+pi))

	if abs(e_slip)<pi/2:
		r_slip = np.exp(-0.1*abs(e_slip))
	elif (e_slip)>= pi/2:
		r_slip = -np.exp(-0.1*(pi-e_slip))
	else:
		r_slip = -np.exp(-0.1*(e_slip+pi))

	vx = obs['linear_vels_x'][0]
	vy = obs['linear_vels_y'][0]
	v = sqrt(vx*vx + vy*vy)


	e_heading_list.append(e_heading)
	e_slip_list.append(e_slip)
	print("----")
	# print("curr_heading: %s"%obs['poses_theta'][0])
	print("way_slip: %s"%ref_point[4])
	# print("way_heading: %s"%ref_point[2])
	# print("e_heading: %s"%e_heading)
	print("----")

	reward = v*(40*r_dis + 40*r_heading + 20*r_slip)

	if v < 4:
		reward  = reward / 2

	return reward

def parse_csv():
	traj = pd.read_csv('waypoints_beta.csv')
	route = traj.values

	return route

if __name__ == '__main__':
	route = parse_csv()
	main(route)