#====================================================
# Imports.
#====================================================

# Pytorch related imports

import torch
import torch.nn as NNet
import torch.optim as optim
import torch.nn.functional as tFunc
import torchvision.transforms as tTrans

# Ploting related imports
import matplotlib
import matplotlib.pyplot as p

# Imports for Math operations 
import math
import random
import numpy as np

# Misc imports 
import gym
from collections import namedtuple
from itertools import count
from PIL import Image
import time

#====================================================
# Utility functions
#====================================================

# === Plotting related functions to plot proformance of algorithm ======

def window_avg(window, val):

	val = torch.tensor(val, dtype=torch.float)
	if len(val) < window:
		avg = torch.zeros(len(val))  # Do not calculate avrage till number of episodes < window size
	else:
		avg = val.unfold(0,window,1).mean(1).flatten(0)
		avg = torch.cat((torch.zeros(window-1), avg))	 

	return avg.numpy()	


def plot(window, val):
	p.figure(2)
	p.clf()
	p.xlabel('Episode ->')
	p.ylabel('Episode Duration ->')
	p.plot(val)			# episode durations
	avg = window_avg(window, val)
	p.plot()
	p.pause(1) #(0.01) Pause time for each frame 
	print("After ", len(val),"episodes: ", "moving avg: " , avg[-1])

# === Experience related utility function =====

Experience = namedtuple('Experience',('s','act','next_s','rwd'))

def tensors_from_expr(experiences):
	# zip() aggregates elements from elements of experiences tensor 
	# Experience() makes it a one large Experience tuple 
	# cat() concatinates separate tensor (of each element of each experience) in one seprate tensor of each element. 

	b = Experience(*zip(*experiences))
	s  = torch.cat(b.s)
	r  = torch.cat(b.rwd)
	ns = torch.cat(b.next_s)
	a  = torch.cat(b.act)
	return (s,a,ns,r)


# =========================================================
# REQUIRED CLASSES 
# =========================================================


# Environment Manager class
class EnvMgr():
	def __init__(self, dev):
		self.e = gym.make('CartPole-v0') #.unwrapped
		self.e.reset()
		self.curr_screen = None
		self.epsd_over = False
		self.cpu_gpu   = dev

	# === FUNCTIONS to get STATE and execute ACTION ========	

	def act(self, action):
		_,rwd, self.epsd_over, _ = self.e.step(action.item())
		return torch.tensor([rwd], device=self.cpu_gpu)

	def state(self):		# return current state of environment in form of processed image
		if self.curr_screen	is None or self.epsd_over:		# start of game or an episode 
			self.curr_screen = self.get_processed_screen()
			blk_screen = torch.zeros_like(self.curr_screen)
			return blk_screen
		else:
			s1 = self.curr_screen
			s2 = self.get_processed_screen()
			self.curr_screen = s2
			return s2 - s1	


	# === FUNCTIONS FOR PROCESSING SCREEEN PIXELS ===

	def render(self, mode='human'):
		return self.e.render(mode)

	def crop(self, screen):
		screen_h = screen.shape[1]
		#remove top and botton pixels of screen
		t = int(screen_h * 0.4)
		b = int(screen_h * 0.8)
		screen = screen[:,t:b,:]
		return screen

	def transform_screen_data(self, screen):
		screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
		screen = torch.from_numpy(screen)

		resize = tTrans.Compose([tTrans.ToPILImage(), tTrans.Resize((40,90)), tTrans.ToTensor()])
		return resize(screen).unsqueeze(0).to(self.cpu_gpu)

	def get_processed_screen(self):
		screen = self.render('rgb_array').transpose((2,0,1))
		screen = self.crop(screen)
		return self.transform_screen_data(screen)

	def screen_w(self):
		screen = self.get_processed_screen()
		return screen.shape[2]

	def screen_h(self):
		screen = self.get_processed_screen()
		return screen.shape[3]

	def close(self):
		self.e.close()

	def reset(self):
		self.e.reset()
		self.curr_screen = None


# Replay Memory class
class replayMem():
	def __init__(self,cap):
		self.mem = []		# holds stored experiences
		self.cap = cap
		self.expr_cnt = 0

	def getRandomExprBatch(self,batch_sz):
		return random.sample(self.mem,batch_sz)

	def isSampalable(self, batch_sz):
		return len(self.mem) >= batch_sz	
		
	def AddExprToMem(self, experience): 
		if len(self.mem) < self.cap:
			self.mem.append(experience)
		else:
			self.mem[ self.expr_cnt % self.cap ] = experience # update oldest experience

		self.expr_cnt += 1


# DQN Network
class DeepQNet(NNet.Module): # Module is base class for all Neural Net modules
	def __init__(self, ip_h, ip_w): # input image height and wiegth 
		super().__init__()

		# NNet.Linear(num of input features, num of output features)
		self.first_full_conn_layer = NNet.Linear(ip_h*ip_w*3, 24) # 3 color channels
		self.second_full_conn_layer = NNet.Linear(24, 32)
		self.output_layer = NNet.Linear(32,2)

	def forward(self, img_tensor): # Mendatory function to implement in a neural network.
		img_tensor = img_tensor.flatten()
		img_tensor = tFunc.relu(self.first_full_conn_layer(img_tensor))
		img_tensor = tFunc.relu(self.second_full_conn_layer(img_tensor))
		img_tensor = self.output_layer(img_tensor)
		return img_tensor


# Exploration - Exploitation 
class EGS():	# Epsilon Greedy Strategy 
	def __init__(self, s, e, d):
		self.decay = d
		self.start = s
		self.end   = e

	def exp_rate(self, curr_step):
		return self.end + (self.start - self.end) * math.exp(-1 * curr_step * self.decay)


# Agent class
class RLAgent():
	def __init__(self, strategy, num_acts, dev): 
		self.strategy  = strategy
		self.curr_step = 0
		self.cpu_gpu   = dev
		self.num_acts   = num_acts 		# How many actions agent can take from a given state. 

	def choose_act(self, st, nn_policy):
		rate = self.strategy.exp_rate(self.curr_step)
		self.curr_step += 1

		if rate > random.random():							# random number between 0 - 1
			a = random.randrange(self.num_acts) 			# explore # random number selected from num_acts random number between 0 - 1
			t = torch.tensor([a]).to(self.cpu_gpu)
			return t

		else:
			with torch.no_grad():			# turn off gradient tracking
				a = nn_policy(st).argmax()	# exploit # nn_policy's highest Q value -> action
				t = torch.tensor([a]).to(self.cpu_gpu)
				return t 


# class for accessing and calculating Q-values
class Qval():
	if torch.cuda.is_available():
		dev = "cuda"
	else:
		dev = "cpu"	

	d 		= torch.device(dev)

	@staticmethod
	def curr_state(pnet,states, actions):
		a = actions.unsqueeze(dim=-1)
		return pnet(states).gather(dim=1, index=a)

	# for each next state, we want to optain max q value pridected by target net among possible next actions. 	
	@staticmethod
	def next_state(tnet,nstates):
		loc_final_s 	= nstates.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
		non_loc_final_s	= (loc_final_s == False)
		batch_sz = nstates.shape[0]
		vals = torch.zeros(batch_sz).to(Qval.d)
		vals[non_loc_final_s] = tnet(non_loc_final_s).max(dim=1)[0].detach()
		return vals


# ===========================================================
# 				MAIN PROGRAM 
# ===========================================================

# === HYPER-PARAMETERS ===== 

mem_size = 100000	# replay memory capacity
batch_sz = 256		
lrn_rt   = 0.001	# policy net learning
dis_rt   = 0.999
target_update = 10	# number of episodes we'll wait before updating target network weights with policy network weights. 

num_epsds = 1000
eps_s 	  = 1		# exploration rate
eps_e     = 0.01
eps_d     = 0.0001

# INITIALIZATIONS ==========

if torch.cuda.is_available():
	dev = "cuda"
else:
	dev = "cpu"	

d 		= torch.device(dev)
cpem 	= EnvMgr(d)

strtgy 	= EGS(eps_s, eps_e, eps_d)
ag 		= RLAgent(strtgy, cpem.e.action_space.n,d)
mem 	= replayMem(mem_size)

# Init both policy DQN and Target DQN with random weights.
Net_policy = DeepQNet(cpem.screen_h(), cpem.screen_w()).to(d)
Net_target = DeepQNet(cpem.screen_h(), cpem.screen_w()).to(d)

#clone policy net in target net
Net_target.load_state_dict(Net_policy.state_dict()) # load policy net model(weights and biases) to target net
Net_target.eval() # This network is not in traning mode. It will only be used for inference.
optm = optim.Adam(Net_policy.parameters(), lrn_rt)

# === TRAINING ===============================

epsd_dur = [] # duration of each episode.

for epsd in range(num_epsds): 	# For each episode
	cpem.reset()
	st = cpem.state() 			# init starting state
	for ts in count():			# for each time step
		# agent uses policy net to decide if it will choose 
		# action based on exploration or exploitation
		a   = ag.choose_act(st, Net_policy) 
		rwd = cpem.act(a)
		next_st = cpem.state()
		mem.AddExprToMem(Experience(st,a,next_st,rwd))
		st = next_st

		if mem.isSampalable(batch_sz):	
			exprns = mem.getRandomExprBatch(batch_sz)
			sts, acts, next_sts, rwds = tensors_from_expr(exprns) 
			
			curr_qvals = Qval.curr_state(Net_policy, sts, acts)
			next_qvals = Qval.next_state(Net_target, next_sts)
			target_qvals = (next_qvals*dis_rt) + rwds

			# mse = mean squared error loss function
			loss = tFunc.mse_loss(curr_qvals, target_qvals.unsqueeze(1)) 
			optm.zero_grad()  # set gradients of all wts and biases = 0
			# Backpropegation: 
			# 1. computes gradient of loss wrt all wts and biases in poliy net
			loss.backward()
			# 2. update wts and biases of poliy net with camputed gradients.
			optm.step()

		if cpem.epsd_over:
			print('episode time: ', ts)
			epsd_dur.append(ts)
			plot(100, epsd_dur)
			break

	if epsd % target_update == 0: # update taeget net
		Net_target.load_state_dict(Net_policy.state_dict())

cpem.close()
