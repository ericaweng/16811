import numpy as np


def initialize_order():
	'''how you initialize the state/structure'''

def SA(points, init_order, initial_temp, final_temp, alpha):
	def propose(args):
		'''define how each proposal step of SA should change 
		the structure/state of the system'''
		pass

	current_temp = initial_temp
	old_score = score(points, init_order)
	curr_order = init_order
	N = init_order.shape[0]
	all_orders = [init_order]  # list of all structures you passed through in the SA run

	# do SA
	while current_temp > final_temp:
		# new structure / state of the system
		new_order = propose(curr_order, N//5)
		new_score = score(points, new_order)
		cost_diff = old_score - new_score

		# accept new structure/state if the cost is lower;
		# or, with some probability
		if cost_diff > 0 or np.random.uniform(0, 1) < np.exp(cost_diff / current_temp):
			curr_order = new_order
			all_orders.append(new_order)
		
		# decrement the temperature			
		current_temp -= alpha  # can replace with multiplicative decrease if you wish

	return curr_order, all_orders

def score(args):
	'''define cost function(lower is better)'''
	pass

np.random.seed(0)