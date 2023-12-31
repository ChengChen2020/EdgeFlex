import os
import logging
import numpy as np


global AwgnErrorTableLdpc1458


# def get_mcs(snr, target_per):
#     assert target_per >= 0 and target_per <= 1
#     snr = max(-1.5, snr)
#     snr = min(30.5, snr)
#
#     for i in range(len(AwgnErrorTableLdpc1458) - 1, -1, -1):
#         print(i)
#         min_snr = AwgnErrorTableLdpc1458[i][0][0]
#         max_snr = AwgnErrorTableLdpc1458[i][-1][0]
#         if snr > min_snr and snr < max_snr:
#             for (d_snr, d_per) in reversed(AwgnErrorTableLdpc1458[i]):
#                 if abs(d_snr - snr) < 0.125 and d_per < target_per:
#                     return i, d_per
#
#     return -1, -1

def setup_logger(logger_name, exp_path, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s %(levelname)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(exp_path, 'output.log'), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)
    return l


def get_statistics():
    snr_set = set()
    for i in range(len(AwgnErrorTableLdpc1458)):
        snr = [a for a, _ in AwgnErrorTableLdpc1458[i]]
        snr_set.update(snr)
    test_list = list(snr_set)
    mean = sum(test_list) / len(test_list)
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
    res = variance ** 0.5
    return mean, res


def get_mcs(snr):
    snr = max(-0.5, snr)
    snr = min(30.5, snr)

    for i in range(len(AwgnErrorTableLdpc1458) - 1, -1, -1):
        max_pre_snr = AwgnErrorTableLdpc1458[i - 1][-1][0] if i >= 1 else -0.5
        min_snr = AwgnErrorTableLdpc1458[i][0][0]
        max_snr = AwgnErrorTableLdpc1458[i][-1][0]
        if snr >= min_snr and snr <= max_snr:
            for (d_snr, d_per) in reversed(AwgnErrorTableLdpc1458[i]):
                if abs(d_snr - snr) < 0.125:
                    return i, d_per
        elif snr >= (min_snr + max_pre_snr) / 2:
            return i, AwgnErrorTableLdpc1458[i][0][1]
        elif snr > max_pre_snr and snr < (min_snr + max_pre_snr) / 2:
            return i - 1, AwgnErrorTableLdpc1458[i - 1][-1][1]
    assert False


# Mbps
def get_data_rate(bandwidth, mcs, per):
    modulation, coding_rate = MCS_mapping[mcs]
    return bandwidth * modulation * coding_rate * (1 - per)


AwgnErrorTableLdpc1458 = [
    # MCS-0
    [
        # [-1.50000, 1.00000],
        # [-1.25000, 0.97950],
        # [-1.00000, 0.60480],
        # [-0.75000, 0.17050],
        [-0.50000, 0.03320],
        [-0.25000, 0.00530],
        [0.00000, 0.00085],
        [0.25000, 0.00022],
        [0.50000, 0.00004],
        [0.75000, 0.00000],
    ],
    # MCS-1
    [
        # [1.50000, 1.00000],
        # [1.75000, 0.97470],
        # [2.00000, 0.62330],
        # [2.25000, 0.18590],
        [2.50000, 0.03400],
        [2.75000, 0.00550],
        [3.00000, 0.00083],
        [3.25000, 0.00015],
        [3.50000, 0.00003],
        [3.75000, 0.00000],
    ],
    # MCS-2
    [
        # [4.00000, 1.00000],
        # [4.25000, 0.98720],
        # [4.50000, 0.62560],
        # [4.75000, 0.15800],
        [5.00000, 0.02090],
        [5.25000, 0.00250],
        [5.50000, 0.00034],
        [5.75000, 0.00003],
        [6.00000, 0.00000],
    ],
    # MCS-3
    [
        # [6.75000, 1.00000],
        # [7.00000, 0.99800],
        # [7.25000, 0.94340],
        # [7.50000, 0.57890],
        # [7.75000, 0.20640],
        [8.00000, 0.04840],
        [8.25000, 0.00930],
        [8.50000, 0.00180],
        [8.75000, 0.00040],
        [9.00000, 0.00011],
        [9.25000, 0.00002],
        [9.50000, 0.00000],
    ],
    # MCS-4
    [
        # [10.00000, 1.00000],
        # [10.25000, 0.99310],
        # [10.50000, 0.70890],
        # [10.75000, 0.24720],
        [11.00000, 0.04700],
        [11.25000, 0.00590],
        [11.50000, 0.00091],
        [11.75000, 0.00016],
        [12.00000, 0.00003],
        [12.25000, 0.00000],
    ],
    # MCS-5
    [
        # [14.00000, 1.00000],
        # [14.25000, 0.99700],
        # [14.50000, 0.91830],
        # [14.75000, 0.53790],
        # [15.00000, 0.16610],
        [15.25000, 0.03690],
        [15.50000, 0.00650],
        [15.75000, 0.00100],
        [16.00000, 0.00031],
        [16.25000, 0.00005],
        [16.50000, 0.00000],
    ],
    # MCS-6
    [
        # [15.50000, 1.00000],
        # [15.75000, 0.98140],
        # [16.00000, 0.73930],
        # [16.25000, 0.33110],
        [16.50000, 0.08150],
        [16.75000, 0.01620],
        [17.00000, 0.00270],
        [17.25000, 0.00052],
        [17.50000, 0.00005],
        [17.75000, 0.00003],
        [18.00000, 0.00000],
    ],
    # MCS-7
    [
        # [17.00000, 1.00000],
        # [17.25000, 0.97750],
        # [17.50000, 0.73980],
        # [17.75000, 0.33190],
        [18.00000, 0.09640],
        [18.25000, 0.02180],
        [18.50000, 0.00470],
        [18.75000, 0.00087],
        [19.00000, 0.00018],
        [19.25000, 0.00003],
        [19.50000, 0.00000],
    ],
    # MCS-8
    [
        # [20.50000, 1.00000],
        # [20.75000, 0.99500],
        # [21.00000, 0.89700],
        # [21.25000, 0.56270],
        # [21.50000, 0.20920],
        [21.75000, 0.05600],
        [22.00000, 0.01170],
        [22.25000, 0.00250],
        [22.50000, 0.00038],
        [22.75000, 0.00013],
        [23.00000, 0.00004],
        [23.25000, 0.00001],
        [23.50000, 0.00000],
    ],
    # MCS-9
    [
        # [22.25000, 1.00000],
        # [22.50000, 0.99900],
        # [22.75000, 0.94080],
        # [23.00000, 0.63600],
        # [23.25000, 0.27190],
        [23.50000, 0.08700],
        [23.75000, 0.02210],
        [24.00000, 0.00500],
        [24.25000, 0.00110],
        [24.50000, 0.00032],
        [24.75000, 0.00004],
        [25.00000, 0.00000],
    ],
    # MCS-10
    [
        # [25.75000, 1.00000],
        # [26.00000, 0.94970],
        # [26.25000, 0.68660],
        # [26.50000, 0.32940],
        # [26.75000, 0.11620],
        [27.00000, 0.03440],
        [27.25000, 0.00880],
        [27.50000, 0.00210],
        [27.75000, 0.00054],
        [28.00000, 0.00009],
        [28.25000, 0.00002],
        [28.50000, 0.00000],
    ],
    # MCS-11
    [
        # [27.75000, 1.00000],
        # [28.00000, 0.94880],
        # [28.25000, 0.75260],
        # [28.50000, 0.40230],
        # [28.75000, 0.16210],
        [29.00000, 0.05150],
        [29.25000, 0.01310],
        [29.50000, 0.00360],
        [29.75000, 0.00100],
        [30.00000, 0.00022],
        [30.25000, 0.00006],
        [30.50000, 0.00000],
    ],
]

# https://en.wikipedia.org/wiki/Wi-Fi_6
# Modulation type, Coding rate
MCS_mapping = [
    (1, 0.5),  # 0
    (2, 0.5),  # 1
    (2, 0.75),  # 2
    (4, 0.5),  # 3
    (4, 0.75),  # 4
    (6, 0.67),  # 5
    (6, 0.75),  # 6
    (6, 0.83),  # 7
    (8, 0.75),  # 8
    (8, 0.83),  # 9
    (10, 0.75),  # 10
    (10, 0.83),  # 11
]

if __name__ == '__main__':
    # mean, res = get_statistics()
    #
    # for i in range(10):
    #     snr = np.random.normal(mean, res)
    #     print(snr, get_mcs(snr))
    #
    # # print(get_data_rate(20, 11))
    #
    import matplotlib.pyplot as plt

    snr_list = list(np.arange(-0.5, 30.5, .1))
    data_rate_list = [get_data_rate(20, get_mcs(snr)[0], get_mcs(snr)[1]) for snr in snr_list]
    plt.title('SNR to Data Rate for 20MHz channel')
    plt.plot(snr_list, data_rate_list)
    plt.xlabel('SNR(dB)')
    plt.ylabel('Data Rate(Mbps)')
    plt.savefig('snr_to_data_rate_mapping.png')
    # mcs, per = get_mcs(30)
    # print(mcs, per)
    # data_rate = get_data_rate(25, mcs, per)
    # print(data_rate)
