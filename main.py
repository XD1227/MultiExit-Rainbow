import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from env import PanicEnv, img_dim, Scenario
import os
from tensorboardX import SummaryWriter

from common.utils import create_log_dir, print_args, set_global_seeds
from common.wrappers import make_atari, wrap_atari_dqn
from arguments import get_args
from train import train
from test import test


def main():
    args = get_args()
    args.noisy = True
    args.double = True
    args.dueling = True
    args.prioritized_replay = True
    args.c51 = True
    args.multi_step = 3
    args.load_agents = True
    args.num_agents = 12
    args.read_model = None
    args.evaluate = False
    print_args(args)

    log_dir = create_log_dir(args)
    if not args.evaluate:
        writer = SummaryWriter(log_dir)

    env = PanicEnv(num_agents=args.num_agents,
                   scenario_=Scenario.Two_Exits, load_agents=True, read_agents=False)

    set_global_seeds(args.seed)

    if args.evaluate:
        test(env, args)
        return

    train(env, args, writer)

    writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))
    writer.close()


if __name__ == "__main__":
    main()
