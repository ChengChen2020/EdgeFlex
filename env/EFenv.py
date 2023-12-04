import pickle
import numpy as np
import matplotlib.pyplot as plt

from helper import get_mcs, get_data_rate

# Worker, AP, Master

# encourage collaboration

# 100MHz

# SNR = gaussian(mean=25)

# throughput = bandwidth * MCS * coding rate * (1 - loss)

# 3 * 32 * 32
# 32 * 16 * 16 * 4 * 8 bits [32KB] [20Mbps] [~13ms]
# n_parts(8) * 16 * 16 * 12 bits [~11]
# 32 * 16 * 16 * 8 bits [4]

# torch arrays GPU-->CPU, packets

# PP (3, 5, 8), NP(1, 2, 4, 8), Inference time , Data size, Accuracy list (AGX, TX2, NX)
# 3 * 4 * 3 = 36
# 5: 1: [4.392, 4.166, 10.449], 32 * 16 * 16 * 4 * 8,
#    8: [],
# 3: 1: [], 24 * 16 * 16 * 4 * 8, 1(8) * 16 * 16 * 12
# 8: 1: [], 64 * 8 * 8 * 4 * 8, 1(8) * 8 * 8 * 12

# Data Size in Bytes
size_dict = {
    3: 16 * 16 / 8,
    5: 16 * 16 / 8,
    8: 8 * 8 / 8
}

# point, embed, part
with open('./env/accuracy.pkl', 'rb') as f:
    accuracy_dict = pickle.load(f)

# device, point, embed, part
# encoding, quantization, decoding
with open('./env/latency.pkl', 'rb') as f:
    latency_dict = pickle.load(f)


def action(states, agent):
    return agent.select_action(states)


class Master:
    def __init__(self, poisson_lambda, sla, device, test=False):
        self.left_task_num = np.random.poisson(poisson_lambda)
        self.left_task_num = max(1, self.left_task_num)
        self.total_task = self.left_task_num
        self.sla = sla
        self.device = device
        self.test = test
        self.wl = 20
        self.enc = []
        self.dec = []
        self.quant = []

    def statistic_init(self):
        self.time_used = 0
        self.finished_num = 0

    def start_task(self):
        """start a new task"""
        if self.left_task_num == 0:
            raise RuntimeError('No tasks left')
        self.encoding_latency, self.quantization_latency, self.decoding_latency = latency_dict(
            (self.device, self.point, self.embed, self.part))

    def action_init(self):
        if self.test:
            self.point = 5
            self.part = 1
            self.embed = 2048
            self.involve = 15  # [1,1,1,1]
        else:
            self.point = np.random.choice([3, 5, 8])
            self.part = np.random.choice([1, 2, 4, 8])
            self.embed = np.random.choice([1024, 2048, 4096])
            self.involve = np.random.choice(np.arange(15))

        return self.point, self.part, self.embed, self.involve

    def working_load(self):
        self.wl = self.wl - np.random.poisson(10) + np.random.poisson(10)
        return self.wl


class Worker:
    def __init__(self, device=0):
        self.curr = 0
        self.offload_size = np.random.normal(10000, 500)
        self.involve = False
        self.device = device
        self.wl = 20
        self.trans = []
        self.dec = []

    def working_load(self):
        self.wl = self.wl - np.random.poisson(10) + np.random.poisson(10)
        return self.wl


class Env:
    def __init__(self, sla, poisson_lambda, num_users=5, beta=0.1, total_band=100, slot_time=500, test=False):
        e_devices = {0: 'AGX', 1: 'TX2', 2: 'NX'}
        self.users = np.random.choice(np.arange(2), num_users)
        self.num_users = num_users
        self.num_workers = num_users - 1
        self.workers = [Worker(self.users[i]) for i in range(1, num_users)]
        self.master = Master(poisson_lambda, sla, self.users[0], test=test)
        self.slot_time = slot_time
        self.total_band = total_band
        self.beta = beta

    def schedule_pf(self):
        for u in self.workers:
            # Current slot throughput / History throughput
            u.weight = u.curr / u.offload_size
        sum_weights = sum([u.weight for u in self.workers if u.involve])
        for u in self.workers:
            if u.involve:
                u.band = self.total_band * u.weight / sum_weights
        # print([u.band for u in self.workers if u.involve])

    def step(self, action):

        point, part, embed, involve = action

        data_size = size_dict[point] * part * np.log2(embed)
        involve_vec = f'{involve:04b}'
        involve_sum = sum(map(int, involve_vec))

        tct = 0.0
        acc = accuracy_dict[(point, embed, part)][involve_sum]
        master_enc, master_quant, master_dec = latency_dict[(self.master.device, point, embed, part)]
        master_working_coeff = 1 + self.master.working_load() / 100

        master_enc *= master_working_coeff
        master_quant *= master_working_coeff
        master_dec *= master_working_coeff

        self.master.enc.append(master_enc)
        self.master.quant.append(master_quant)
        self.master.dec.append(master_dec)

        tct += master_enc + master_quant

        for i in range(self.num_workers):
            if involve_vec[i] == '1':
                self.workers[i].involve = True
                self.workers[i].curr = data_size
            else:
                self.workers[i].involve = False

        self.schedule_pf()

        for u in self.workers:
            if u.involve:
                _, _, dec = latency_dict[(u.device, point, embed, part)]
                u.offload_size += data_size
                # u.dec = dec * (1 + u.working_load() / 100)
                snr = np.random.normal(25, 5)
                mcs, per = get_mcs(snr)
                data_rate = get_data_rate(u.band, mcs, per)
                # u.trans = data_size / data_rate

                u.dec.append(dec * (1 + u.working_load() / 100))
                u.trans.append(data_size / data_rate)
            else:
                u.dec.append(0)
                u.trans.append(0)

        dec_trans = max([u.dec[-1] + u.trans[-1] for u in self.workers] + [master_dec])
        tct += dec_trans

        # print(acc - self.master.sla)
        # print(self.beta * tct, master_enc, master_quant, dec_trans)

        # state = self.get_state()
        reward = (acc - self.master.sla) / 10. - self.beta * tct / 100.

        self.master.left_task_num -= 1

        done = True if self.is_done() else False

        # DIM = 23
        state = []
        for u in self.workers:
            state.append(u.dec[-1])
            state.append(u.trans[-1])
            state.append(np.mean(u.dec))
            state.append(np.mean(u.trans))
        state.extend([master_enc, master_quant, master_dec])
        state.extend([np.mean(self.master.enc), np.mean(self.master.quant), np.mean(self.master.dec)])
        state.append(self.master.sla)

        info = {
            'acc_advantage': acc - self.master.sla,
            'task_completion_time': tct
        }

        return state, reward, done, info

    def is_done(self):
        if self.master.left_task_num == 0:
            return True
        return False


if __name__ == '__main__':
    ed = np.random.choice(np.arange(2), 5)
    master_params = {
        'poisson_lambda': 200,
        'sla': 75
    }
    worker_params = {
        'device': 0
    }
    # Env(500, 5, **master_params, **worker_params)
    # somelists = [
    #     [5, 3, 8],
    #     [1024, 2048, 4096],
    #     [1, 2, 4, 8]
    # ]
    #
    #
    # data = {}

    # with open('/Users/chen4384/Desktop/accuracy.txt') as f:
    #     lines = [line.rstrip() for line in f]
    #
    # for i, element in enumerate(itertools.product(*somelists)):
    #     data[element] = tuple(float(x) for x in lines[i].split(','))
    #
    # print(data[(5, 2048, 2)])

    # with open('accuracy.pkl', 'wb') as f:
    #     pickle.dump(data, f)

    # with open('accuracy.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
    # reward_list = []
    # tct_list = []
    # acc_list = []
    # max_num_episodes = 1
    # for episode in range(max_num_episodes):
    #     env = Env(sla=np.random.choice(np.arange(70, 80)), poisson_lambda=500)
    #     while not env.is_done():
    #         reward, tct, acc, _ = env.step(env.master.action_init())
    #
    #         reward_list.append(reward)
    #         tct_list.append(tct)
    #
    # print(len(reward_list))
    # plt.plot(reward_list)
    # # plt.xlabel('Step')
    # # plt.ylabel('Reward')
    # plt.savefig('Reward.png')



