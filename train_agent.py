import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from agent.ppo import PPO
from env.EFenv import Env
from helper import setup_logger


def train(args, agent, save_dir, logger, start_episode=0, start_step=0):
    global_episode = start_episode
    global_step = start_step
    # max_episode_step = int((args.poisson_lambda * 0.15) / args.slot_time)
    max_episode_step = args.poisson_lambda * 2

    local_reward_list = []
    reward_list = []
    actor_loss = []
    critic_loss = []

    best_reward = 0.0
    while True:
        env = Env(sla=args.sla, poisson_lambda=args.poisson_lambda, beta=args.beta, total_band=100)
        actions = env.master.action_init()
        for j in range(max_episode_step):

            s_t, r_t, done, _, _ = env.step(actions)

            agent.buffer.states.append(s_t)
            agent.buffer.rewards.append(r_t)
            agent.buffer.is_terminals.append(done)

            global_step += 1

            actions = agent.select_action(s_t)

            if global_step % args.step == 0:
                # env.get_users()
                loss_a, loss_c = agent.update()
                avg_reward, local_reward = test(args, global_step, agent, logger)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    agent.save_model(os.path.join(save_dir, 'ckpt.pt'), args)

                reward_list.append(avg_reward)
                local_reward_list.append(local_reward)
                actor_loss.append(loss_a)
                critic_loss.append(loss_c)

            if done:
                global_episode += 1
                break

        if global_step > args.max_global_step:
            # test(args, global_episode, global_step, test_env, agent, logger)
            plt.plot(reward_list, label='EdgeFlex')
            plt.plot(local_reward_list, label='Local')
            plt.xlabel('Episode')
            plt.legend()
            plt.ylabel('Averaged episode reward')
            plt.savefig(os.path.join(save_dir, 'reward_record.jpg'))
            break

    # agent.save_model(save_dir + 'ckp.pt', args)


def test(args, step, agent, logger=None):
    done = False
    # env = Env(sla=np.random.choice(np.arange(70, 80)), poisson_lambda=500, test=True)
    env = Env(sla=args.sla, poisson_lambda=400, test=True, beta=args.beta, total_band=100)
    # env.get_users()
    local_reward = 0.
    test_reward = 0.
    acc_term = 0.
    tct_term = 0.
    finished = 0
    actions = env.master.action_init()
    # master_work_load = []
    while not done:
        # master_work_load.append(env.master.wl)
        s_t, r_t, done, info, l_r_t = env.step(actions)
        test_reward += r_t
        local_reward += l_r_t
        acc_term += info['acc']
        tct_term += info['tct']
        finished += 1
        actions = agent.select_action(s_t, test=True)

    avg_acc_term = acc_term / finished
    avg_tct_term = tct_term / finished
    avg_test_reward = test_reward / finished
    avg_local_reward = local_reward / finished
    if logger is not None:
        logger.info(f'step {step}, reward {avg_test_reward:.4f}, ({avg_acc_term:.6f} {avg_tct_term:.6f})')

    return avg_test_reward, avg_local_reward


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='default_DRL')

    # system
    parser.add_argument('--net', default='mobilenetv2', type=str)
    parser.add_argument('--poisson_lambda', default=500, type=int)
    parser.add_argument('--sla', default=0.9, type=float)
    parser.add_argument('--num_points', default=3, type=int)
    parser.add_argument('--num_parts', default=4, type=int)
    parser.add_argument('--num_embeds', default=3, type=int)
    parser.add_argument('--num_involve', default=16, type=int)
    parser.add_argument('--num_users', default=5, type=int)
    parser.add_argument('--beta', default=1, type=float)

    # PPO
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
    exp_name = f'{args.net}_{args.sla}_{args.beta}_EdgeFlex'
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
        # 'pmax': args.pmax,
        'gamma': args.gamma,
        'lam': args.lam,
        'repeat_time': args.repeat_time,
        'batch_size': args.batch_size,
        'eps_clip': args.eps_clip,
        'w_entropy': args.w_entropy,
    }

    agent = PPO(**agent_params)

    train(args, agent, os.path.join('result', exp_name), logger)
