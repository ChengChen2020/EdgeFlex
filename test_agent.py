import os
import argparse

import torch

from agent.ppo import PPO
from env.EFenv import Env
from helper import setup_logger


def test(agent, logger=None):
    done = False
    env = Env(sla=70, poisson_lambda=500, test=True, beta=args.beta, total_band=100)
    test_reward = 0.
    acc_term = 0.
    tct_term = 0.
    finished = 0
    actions = env.master.action_init()
    while not done:
        # master_work_load.append(env.master.working_load())
        s_t, r_t, done, info = env.step(actions)
        # print(s_t)
        # print(actions)
        test_reward += r_t
        acc_term += info['acc']
        tct_term += info['tct']
        finished += 1
        actions = agent.select_action(s_t, test=True)

    avg_acc_term = acc_term / finished
    avg_tct_term = tct_term / finished
    avg_test_reward = test_reward / finished
    if logger is not None:
        logger.info(f'Reward {avg_test_reward:.4f}, ({avg_acc_term:.6f} {avg_tct_term:.6f})')

    return avg_test_reward


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='default_DRL')

    # system
    parser.add_argument('--net', default='mobilenetv2', type=str)
    parser.add_argument('--poisson_lambda', default=100, type=int)
    parser.add_argument('--num_points', default=3, type=int)
    parser.add_argument('--num_parts', default=4, type=int)
    parser.add_argument('--num_embeds', default=3, type=int)
    parser.add_argument('--num_involve', default=16, type=int)
    parser.add_argument('--num_users', default=5, type=int)
    parser.add_argument('--beta', default=1, type=float)

    parser.add_argument('--lr_a', default=0.0001, type=float, help='actor net learning rate')
    parser.add_argument('--lr_c', default=0.0001, type=float, help='critic net learning rate')

    parser.add_argument('--max_global_step', type=int, default=500000)
    parser.add_argument('--gamma', type=float, default=0.95)
    # parser.add_argument('--slot_time', default=0.5, type=float)

    parser.add_argument('--repeat_time', default=20, type=int)
    parser.add_argument('--step', default=1024, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lam', default=0.95, type=float)
    parser.add_argument('--eps_clip', default=0.2, type=float)
    parser.add_argument('--w_entropy', default=0.001, type=float)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_parser()
    exp_name = f'{args.net}_EdgeFlex'
    os.makedirs(os.path.join('result', exp_name), exist_ok=True)
    logger = setup_logger(__name__, os.path.join('result', exp_name))

    d_args = vars(args)
    for k in d_args.keys():
        logger.info(f'{k}: {d_args[k]}')

    agent_params = {
        'num_users': 5,
        'num_states': 10,
        'num_points': args.num_points,
        'num_parts': args.num_parts,
        'num_embeds': args.num_embeds,
        'num_involve': args.num_involve,
        'lr_a': args.lr_a,
        'lr_c': args.lr_c,
        'gamma': args.gamma,
        'lam': args.lam,
        'repeat_time': args.repeat_time,
        'batch_size': args.batch_size,
        'eps_clip': args.eps_clip,
        'w_entropy': args.w_entropy,
    }

    agent = PPO(**agent_params)
    dic = torch.load('result/mobilenetv2_EdgeFlexckp.pt')
    agent.load_model(dic['actor'], dic['critic'])
    agent.actor.eval()
    agent.critic.eval()
    with torch.no_grad():
        test(agent, logger)
