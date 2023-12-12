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

min_acc = 67.97
max_acc = 79.1

min_enc_quant = 4.707
max_enc_quant = 22.030000

min_trans = 0.22
max_trans = 34.13

min_dec = 6.760
max_dec = 20.045

# Data_Size 80 ~ 3072
# Band 360
min_tct = 15
max_tct = 75


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
        self.wl = 40
        self.enc = []
        self.dec = []
        self.quant = []

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

    def bg_working_load(self):
        self.wl = self.wl - 20 + np.random.poisson(20)
        self.wl = min(60, self.wl)
        self.wl = max(20, self.wl)
        return self.wl


class Worker:
    def __init__(self, device=0):
        self.curr = 0
        self.offload_size = np.random.normal(10000, 500)
        self.involve = False
        self.device = device
        self.wl = 30
        self.trans = []
        self.dec = []
        self.band = 25

    def bg_working_load(self):
        if self.device == 0:
            pw = 20
        elif self.device == 1:
            pw = 10
        else:
            pw = 15
        self.wl = self.wl - pw + np.random.poisson(10)
        self.wl = min(50, self.wl)
        self.wl = max(10, self.wl)
        return self.wl


class Env:
    def __init__(self, sla, poisson_lambda, num_users=5, beta=0.1, total_band=100, slot_time=500, test=False):
        self.e_devices = {0: 'AGX', 1: 'TX2', 2: 'NX'}
        # self.users = np.random.choice(np.arange(2), num_users)
        self.users = [0, 1, 1, 2, 0]
        self.num_users = num_users
        self.num_workers = num_users - 1
        self.workers = [Worker(self.users[i]) for i in range(1, num_users)]
        self.master = Master(poisson_lambda, sla, self.users[0], test=test)
        self.slot_time = slot_time
        self.total_band = total_band
        self.beta = beta

    def get_users(self):
        for u in self.users:
            print(self.e_devices[u], end=',')
        print()

    def schedule_pf(self):
        for u in self.workers:
            # Current slot throughput / History throughput
            u.weight = u.curr / u.offload_size
        sum_weights = sum([u.weight for u in self.workers if u.involve])
        for u in self.workers:
            if u.involve:
                u.band = self.total_band * u.weight / sum_weights
        print([u.band for u in self.workers if u.involve])

    def step(self, action):

        point, part, embed, involve = action

        data_size = size_dict[point] * part * np.log2(embed)
        involve_vec = f'{involve:04b}'
        involve_sum = sum(map(int, involve_vec))

        tct = 0.0
        acc = accuracy_dict[(point, embed, part)][involve_sum]
        master_enc, master_quant, master_dec = latency_dict[(self.master.device, point, embed, part)]
        self.master.bg_working_load()
        self.master.wl += point
        self.master.wl = min(100, self.master.wl)
        master_working_coeff = 1 + self.master.wl / 100.

        master_enc *= master_working_coeff
        master_quant *= master_working_coeff
        master_dec *= master_working_coeff

        self.master.enc.append(master_enc)
        self.master.quant.append(master_quant)
        self.master.dec.append(master_dec)

        if involve_sum > 0:
            tct += master_enc + master_quant
        else:
            tct += master_enc

        for i in range(self.num_workers):
            if involve_vec[i] == '1':
                self.workers[i].involve = True
                self.workers[i].curr = data_size
            else:
                self.workers[i].involve = False

        # self.schedule_pf()

        for u in self.workers:
            snr = np.random.normal(25, 5)
            mcs, per = get_mcs(snr)
            u.mcs = mcs
            if u.involve:
                _, _, dec = latency_dict[(u.device, point, embed, part)]

                u.band = self.total_band / involve_sum
                u.data_rate = get_data_rate(u.band, mcs, per)

                u.bg_working_load()
                u.wl += 22 - point
                u.wl += part * embed / 1024
                u.wl = min(100, u.wl)

                u.dec.append(dec * (1 + u.wl / 100))
                u.trans.append(data_size / u.data_rate)
            else:
                u.bg_working_load()
                u.dec.append(0.)
                u.trans.append(0.)

        dec_trans = max([u.dec[-1] + u.trans[-1] for u in self.workers if u.involve] + [master_dec])
        tct += dec_trans

        # print(acc - self.master.sla)
        # print(self.beta * tct, master_enc, master_quant, dec_trans)

        # state = self.get_state()
        # reward = (acc - self.master.sla) / 10. - self.beta * tct / 100.

        local_acc = 76
        local_tct = master_enc + master_dec

        # print(tct, local_tct)

        acc_term = (acc - min_acc) / (max_acc - min_acc)
        tct_term = (tct - min_tct) / (max_tct - min_tct)

        local_acc_term = (local_acc - min_acc) / (max_acc - min_acc)
        local_tct_term = (local_tct - min_tct) / (max_tct - min_tct)

        reward = 1 if acc_term >= self.master.sla else 0
        if acc_term > self.master.sla:
            reward -= (acc_term - self.master.sla)
        # reward = 1.5 * acc_term
        reward -= tct_term
        # print(reward)

        local_reward = 1 if local_acc_term >= self.master.sla else 0
        if local_acc_term > self.master.sla:
            local_reward -= (local_acc_term - self.master.sla)
        # reward = 1.5 * acc_term
        local_reward -= local_tct_term

        # print(reward, local_reward)

        self.master.left_task_num -= 1

        done = True if self.is_done() else False

        # DIM = 23
        state = []
        for u in self.workers:
            # state.append(u.dec[-1])
            # state.append(u.trans[-1])
            state.append(u.wl / 100.)
            state.append(u.mcs / 11.)
            # non_zero_values = [value for value in u.dec if value != 0]
            # if len(non_zero_values) == 0:
            #     state.append(0)
            # else:
            #     mean_without_zeros = np.mean(non_zero_values)
            #     state.append(mean_without_zeros)
            # state.append(np.mean(u.trans))
        # state.extend([master_enc, master_quant, master_dec])
        state.append(self.master.wl / 100.)
        state.append((data_size - 80.) / (3072. - 80.))
        # state.extend([acc])
        # state.append(data_size)
        # state.append(self.total_band / (involve_sum + 1e-6))

        # print(state)
        # print(action)
        # print(master_enc, master_quant, master_dec, dec_trans, tct)
        # print(acc, tct_term)

        # print(state)
        # state.append(self.master.sla)

        info = {
            'acc': acc,
            'tct': tct
        }

        return state, reward, done, info, local_reward

    def is_done(self):
        if self.master.left_task_num == 0:
            return True
        return False


if __name__ == '__main__':
    mcs, per = get_mcs(30)
    print(mcs, per)
    # ed = np.random.choice(np.arange(2), 5)
    # master_params = {
    #     'poisson_lambda': 200,
    #     'sla': 75
    # }
    # worker_params = {
    #     'device': 0
    # }
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



