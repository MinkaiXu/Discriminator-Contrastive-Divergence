import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str, default='configs/base.yml',
                    help='path to config file')
parser.add_argument('--gpu', type=int, default=0, help='index of gpu to be used')
parser.add_argument('--data_dir', type=str, default='./data/imagenet')
parser.add_argument('--results_dir', type=str, default='./results/gans',
                    help='directory to save the results to')
parser.add_argument('--inception_model_path', type=str, default='./datasets/inception_model',
                    help='path to the inception model')
parser.add_argument('--FID_stat_file', type=str, default='./datasets/fid_stats_cifar10_train.npz',
                    help='path to the FID model')
parser.add_argument('--D_snapshot', type=str, default='',
                    help='path to the snapshot of D')
parser.add_argument('--G_snapshot', type=str, default='',
                    help='path to the snapshot of G')

parser.add_argument('--sampling_space', type=str, default='latent',
                    help="sampling space: 'latent' or 'pixel'")

# Setting for MCMC sampling
parser.add_argument('--num_steps', type=int, default=20,
                    help='Steps of gradient descent for training')
parser.add_argument('--langevin_steps_each', type=int, default=10,
                    help='Langevin Steps of gradient descent for training')
parser.add_argument('--noise_scale', type=float, default=0.005,
                    help='Relative amount of noise for MCMC')
parser.add_argument('--step_lr', type=float, default=1.0,
                    help='Size of steps for gradient descent')
parser.add_argument('--eval_num_steps', type=int, default=-1,
                    help='Steps of evaluation gradient descent for training')
parser.add_argument('--eval_noise_scale', type=float, default=0.005,
                    help='Relative amount of noise for MCMC')
parser.add_argument('--eval_step_lr', type=float, default=1.0,
                    help='Size of evaluation steps for gradient descent')
parser.add_argument('--temperature', type=int, default=1,
                    help='Temperature for energy function')
parser.add_argument('--proj_norm', type=float, default=1.0,
                    help='Maximum change of input images')
parser.add_argument('--proj_norm_type', type=str, default='li',
                    help='Either li or l2 ball projection')
parser.add_argument('--hmc', type=bool, default=False,
                    help='Whether to use HMC sampling to train models')

parser.add_argument('--snapshot', type=str, default='',
                    help='path to the snapshot')
parser.add_argument('--loaderjob', type=int,
                    help='number of parallel data loading processes')


# Settings for sampling
parser.add_argument('--splits', type=int, default=10)
parser.add_argument('--tf', action='store_true', default=False)
parser.add_argument('--anealing', action='store_true', default=False)
parser.add_argument('--auto_noise', action='store_true', default=False)

# Generating Images settings
parser.add_argument('--rows', type=int, default=5)
parser.add_argument('--columns', type=int, default=5)
parser.add_argument('--classes', type=int, nargs="*", default=None)

args = parser.parse_args()

if args.eval_num_steps == -1:
    args.eval_num_steps = args.num_steps

noise_scale_name='auto' if args.auto_noise else str(args.noise_scale)
eval_noise_scale_name='auto' if args.auto_noise else str(args.eval_noise_scale)

# args.results_dir = args.results_dir + '/' + \
#     'num_steps' + str(args.num_steps) + \
#     'eval_num_steps' + str(args.eval_num_steps) + \
#     'noise_scale' + noise_scale_name + \
#     'eval_noise_scale' + eval_noise_scale_name + \
#     'step_lr' + str(args.step_lr) + \
#     'eval_step_lr' + str(args.eval_step_lr) + \
#     'anearling' + str(args.anealing)
args.results_dir = args.results_dir + '/' + \
    'eval_num_steps' + str(args.eval_num_steps) + \
    'eval_noise_scale' + eval_noise_scale_name + \
    'eval_step_lr' + str(args.eval_step_lr) + \
    'anearling' + str(args.anealing)
args.eval_results_dir = args.results_dir
