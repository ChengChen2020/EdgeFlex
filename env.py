import numpy as np

### Worker, AP, Master

# encourage collaboration

# 100MHz

# SNR = gaussian(mean=25)

# throughput = bandwidth * MCS * coding rate * (1 - loss)

# 3 * 32 * 32
# 32 * 16 * 16 * 4 * 8 bits [32KB] [20Mbps] [~13ms]
# n_parts(8) * 16 * 16 * 12 bits [~11]
# 32 * 16 * 16 * 8 bits [4]

# torch arrays GPU-->CPU, packets

# [NX, AGX]

edevices = {0:'AGX', 1:'TX2', 2:'NX'}
ed = np.random.choice(np.arange(2), 5)



# PP (3, 5, 8), NP(1, 2, 4, 8), Inference time , Data size, Accuracy list (AGX, TX2, NX)
# 3 * 4 * 3 = 36
5: 1: [4.392, 4.166, 10.449], 32 * 16 * 16 * 4 * 8, 
   8: [],
3: 1: [], 24 * 16 * 16 * 4 * 8, 1(8) * 16 * 16 * 12
8: 1: [], 64 * 8 * 8 * 4 * 8, 1(8) * 8 * 8 * 12


class Master:
	def __init__(self, possion_lambda, sla):
		self.left_task_num = np.random.poisson(self.possion_lambda)
		self.left_task_num = max(1, self.left_task_num)
		self.total_task = self.left_task_num
		self.sla = sla

	def statistic_init(self):
		self.time_used = 0
		self.finished_num = 0

	def start_task(self):
		'''start a new task'''
		if self.left_task_num == 0:
		   raise RuntimeError('No tasks left')
		data_size, infer_latency = get_data(self.point, self.part, self.device)
		self.time_left = infer_latency
		self.data_left = data_size


	def action_init(self):
		if self.test:
			self.point = 5
			self.part = 1
			self.involve = 15 # [1,1,1,1]
		else:
			self.point = np.random.choice(np.arange(10))
			self.part = np.random.choice([1, 2, 4, 8])
			self.involve = np.random.choice(np.arange(15))


class Worker:
	def __init__(self, snr, device=0):
		self.decode_latency = get_data(self.device, self.point)

	def get_states(self):
		self.snr = np.random.normal(25, 5)
		self.wl = np.random.poisson(20)
		self.decode_latency *= 1 + self.wl / 100
		return self.snr, self.decode_latency


class Env:
	def __init__(self, slot_time, num_users, worker_params, master_params, beta=0.5, total_band=100):
		self.num_users = num_users
		self.workers = [Worker(**worker_params) for _ in range(num_users)]
		self.master = Master(**master_params)
		self.slot_time = slot_time
		self.total_band = total_band
		self.beta = beta

	def schedule_init(self):
		for u in self.workers:
			u.band = self.total_band / self.num_users

	def schedule_pf(self):
		for u in self.workers:
			# Current slot throughput / History throughput
			u.weight = u.curr / u.hist
		sum_weights = [u.weight for u in self.workers if u.involve]
		for u in self.workers:
			if u.involve:
				u.band = self.total_band * u.weight / sum_weights

	def offload_latency(self):
		# infer_latency
		data_size, encode_latency = get_data(self.point, self.part)
		for u in self.workers:
			if u.involve:
				snr, decode_latency = u.get_states()
				# bandwidth_Hz * modulation_efficiency * coding_rate * (1 - loss_rate)
				data_rate = u.band * 6 * 0.75 * (1 - 0.001 * (10 ** (-snr / 10))) ** (1500 * 8)
				u.offload_latency = data_size / data_rate


	def get_states(self):
		state = []
		for u in self.workers:
			if u.involve:
				state.append(u.decode_latency)
				state.append(u.offload_latency)
			else:
				state.append()
		return state

	def get_rewards(self):
		energy = np.mean([u.energy_used for u in self.UEs])
		finished = np.mean([u.finished_num for u in self.UEs])
		avg_e = energy / max(finished, 0.8)
		avg_t = self.slot_time / max(finished, 0.8)
		reward = self.master.sla - acc - self.beta * tct
		return reward

	def step(self, action):
		if self.is_done():
            done = True

		state = self.get_state()
        reward = self.get_reward()

        return state, reward, done, info

    def is_done(self):
    	if self.master.left_task_num == 0:
    		return True
    	return False







