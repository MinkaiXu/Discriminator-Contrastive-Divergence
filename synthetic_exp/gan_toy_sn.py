import os, sys

sys.path.append(os.getcwd())

import random
import argparse
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
from mcmc_sampler import Langevin_dynamics
import tflib as lib
import tflib.plot
import pdb

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from mcmc_sampler import spectral_norm

# from args import args
# print (args.fintuning)

torch.manual_seed(1)
parser = argparse.ArgumentParser(description='langevin dynamics')

# Finetuning = False

MODE = 'wgan-gp'  # wgan or wgan-gp OR GAN
Output = 'linear'
types = 'linear_128_fintunes_sn_25_10000step'

# DATASET = 'swissroll'  # 8gaussians, 25gaussians, swissroll
DIM = 128  # Model dimensionality
FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
# Gaussian noise, as in the plots in the paper
LAMBDA = .0  # Smaller lambda seems to help for toy tasks specifically
# CRITIC_ITERS = 5# How many critic iterations per generator iteration
BATCH_SIZE = 256  # Batch size
ITERS = 250  # how many generator iterations to train for
Fintune_iters = 1
EPOCHS = 200
Fintune_Epochs = 1
# parser.add_argument('--MODE', type=str, default='wgan-gp', choices=['wgan', 'wgan-gp', 'GAN'])
parser.add_argument('--dataset', type=str, default='8gaussians', help=" 8gaussians, 25gaussians, swissroll")
# parser.add_argument('--frame', type=int, default=33, help="from which to name the result image")
parser.add_argument('--ci', type=int, default=5, help="how much update for d")
parser.add_argument('--steplr', type=float, default=0.2, help="step rate")
parser.add_argument('--step', type=float, default=60, help="mc steps")
parser.add_argument('--noise', type=float, default=0.2, help="mc steps")
parser.add_argument('--fintuning', type=bool, default=False, help="whether funtuning")
args = parser.parse_args()
DATASET = args.dataset
Finetuning = args.fintuning
CRITIC_ITERS = args.ci

use_cuda = True

if not os.path.exists('tmp/' + DATASET + types +'ci' + str(CRITIC_ITERS)):
    os.mkdir('tmp/' + DATASET + types+'ci' + str(CRITIC_ITERS))

log = open('tmp/' + DATASET + types+'ci' + str(CRITIC_ITERS)+'/log.txt','w')

log.write(MODE + DATASET+ str(LAMBDA) + str(CRITIC_ITERS))
log.close()
# print(MODE,DATASET,LAMBDA,CRITIC_ITERS)
# ==================Definition Start======================
def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 2),
        )
        self.main = main

    def forward(self, noise, real_data):
        if FIXED_GENERATOR:
            return noise + real_data
        else:
            output = self.main(noise)
            return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            spectral_norm(nn.Linear(2, DIM)),
            nn.ReLU(True),
            spectral_norm(nn.Linear(DIM, DIM)),
            nn.ReLU(True),
            spectral_norm(nn.Linear(DIM, DIM)),
            nn.ReLU(True),
            spectral_norm(nn.Linear(DIM, 1)),
        )
        self.main = main
    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


adv_loss = torch.nn.BCELoss()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

frame_index = [0]
def generate_image(true_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    BATCHs = 8
    N_POINTS = 128
    RANGE = 3
    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    points_v = autograd.Variable(torch.Tensor(points)).requires_grad_(False)
    if use_cuda:
        points_v = points_v.cuda()
    disc_map = netD(points_v).cpu().data.numpy()
    # pdb.set_trace()
    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.figure(1)
    plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.contourf(x, y, disc_map.reshape((len(x), len(y))).transpose(), alpha=0.1)
    plt.figure(2)
    plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.contourf(x, y, disc_map.reshape((len(x), len(y))).transpose(), alpha=0.1)
    for i in range(BATCHs):
        noise = torch.randn(BATCH_SIZE, 2)
        if use_cuda:
            noise = noise.cuda()
        noisev = autograd.Variable(noise).requires_grad_(True)
        true_dist_v = autograd.Variable(torch.Tensor(true_dist).cuda() if use_cuda else torch.Tensor(true_dist))
        samples = netG(noisev, true_dist_v)
        if not FIXED_GENERATOR:
            plt.figure(1)
            plt.scatter(samples.cpu().data.numpy()[:, 0], samples.cpu().data.numpy()[:, 1], s=20, c='black',
                        edgecolors='none', marker='o')  # alpha=0.1,
        # plt.savefig('tmp/' + DATASET + types + 'ci' + str(CRITIC_ITERS) + '/' + 'frame' + str(frame_index[0]) + '.jpg')

        # samples.requires_grad_(True)
        # samples_new = Langevin_dynamics(samples, netD, n_steps=5000, step_lr=0.0005)
        samples_new = Langevin_dynamics(samples, netD, n_steps=200, step_lr=args.step_lr,noisel=args.noise,train=False)
        # print(samples.size())
        if not FIXED_GENERATOR:
            plt.figure(2)
            plt.scatter(samples_new.data.numpy()[:, 0], samples_new.data.numpy()[:, 1], s=20, c='red',
                        edgecolors='none', marker='o')  # alpha=0.1,

    # plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='o',alpha=0.3,edgecolors='none')
    plt.figure(1)
    plt.savefig('tmp/' + DATASET + types+'ci' + str(CRITIC_ITERS)+ '/' + 'frame_original' + str(frame_index[0]) + '.jpg')
    plt.figure(2)
    plt.savefig('tmp/' + DATASET + types + 'ci' + str(CRITIC_ITERS) + '/' + 'frame_dynamic' + str(frame_index[0]) + '.jpg')
    frame_index[0] += 1

def clip_grad(parameters, optimizer):
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'step' not in state or state['step'] < 1:
                    continue
                step = state['step']
                exp_avg_sq = state['exp_avg_sq']
                _, beta2 = group['betas']
                bound = 3 * torch.sqrt(exp_avg_sq / (1 - beta2 ** step)) + 0.1
                p.grad.data.copy_(torch.max(torch.min(p.grad.data, bound), -bound))




# Dataset iterator
def inf_train_gen():
    if DATASET == '25gaussians':

        dataset = []
        for i in range(int(64000 / 25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += x
                    point[1] += y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        # dataset /= 2.828  # stdev
        while True:
            for i in range(int(len(dataset) / BATCH_SIZE)):
                yield dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    elif DATASET == '20gaussians':

        dataset = []
        for i in range(int(64000 / 25)):
            for x in [-2,-0,1,2]:
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += x
                    point[1] += y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        # dataset /= 2.828  # stdev
        while True:
            for i in range(int(len(dataset) / BATCH_SIZE)):
                yield dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    elif DATASET == 'swissroll':

        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE,
                noise=0.25
            )[0]
            data = data.astype('float32')[:, [0, 2]]
            data /= 7.5  # stdev plus a little
            yield data

    elif DATASET == '8gaussians':

        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in range(BATCH_SIZE):
                point = np.random.randn(2) * .02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev
            yield dataset


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1)-1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================

netG = Generator()
# if Output == 'linear':
#     netD = Discriminator()
# else:
#     netD = Discriminator()

netD = Discriminator()
# netD = Discriminator()
netD.apply(weights_init)
netG.apply(weights_init)
print (netG)
print (netD)

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

data = inf_train_gen()
Tensor = torch.cuda.FloatTensor

if not Finetuning:
    for epoch in range(EPOCHS):
        for iteration in range(ITERS):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            for iter_d in range(CRITIC_ITERS):
                _data = data.__next__()
                real_data = torch.Tensor(_data)
                valid_l = autograd.Variable(Tensor(real_data.size(0), 1).fill_(1.0), requires_grad=False)
                fake_l = autograd.Variable(Tensor(real_data.size(0), 1).fill_(0.0), requires_grad=False)
                if use_cuda:
                    real_data = real_data.cuda()
                real_data_v = autograd.Variable(real_data)
                netD.zero_grad()

                if Output == 'linear':
                    D_real = netD(real_data_v)
                    D_real = D_real.mean()
                    D_real.backward(mone)
                    noise = torch.randn(BATCH_SIZE, 2)
                    if use_cuda:
                        noise = noise.cuda()
                    noisev = autograd.Variable(noise).requires_grad_(False)  # totally freeze netG
                    fake = autograd.Variable(netG(noisev, real_data_v).data)
                    inputv = fake
                    D_fake = netD(inputv)
                    D_fake = D_fake.mean()
                    D_fake.backward(one)
                    # rain with gradient penalty
                    gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
                    gradient_penalty.backward()
                    D_cost = D_fake - D_real + gradient_penalty
                    Wasserstein_D = D_real - D_fake
                    optimizerD.step()
                else:
                    D_real = netD(real_data_v)
                    # D_real = D_real.mean()
                    # D_real.backward(mone)
                    # train with fake
                    noise = torch.randn(BATCH_SIZE, 2)
                    if use_cuda:
                        noise = noise.cuda()
                    noisev = autograd.Variable(noise).requires_grad_(False)  # totally freeze netG
                    fake = autograd.Variable(netG(noisev, real_data_v).data)
                    inputv = fake
                    D_fake = netD(inputv)
                    # train with gradient penalty
                    gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
                    gradient_penalty.backward()
                    real_loss = adv_loss(D_real, valid_l)
                    fake_loss = adv_loss(D_fake, fake_l)
                    d_loss = (real_loss + fake_loss) / 2
                    # D_cost = D_fake - D_real + gradient_penalty
                    D_cost = d_loss
                    D_cost.backward()
                    Wasserstein_D = D_real - D_fake
                    optimizerD.step()
            if not FIXED_GENERATOR:
                ############################
                # (2) Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid computation
                netG.zero_grad()

                _data = data.__next__()
                real_data = torch.Tensor(_data)
                if use_cuda:
                    real_data = real_data.cuda()
                real_data_v = autograd.Variable(real_data)

                noise = torch.randn(BATCH_SIZE, 2)
                if use_cuda:
                    noise = noise.cuda()
                noisev = autograd.Variable(noise)
                fake = netG(noisev, real_data_v)
                G = netD(fake)
                if Output == 'linear':
                    G = netD(fake)
                    G = G.mean()
                    G.backward(mone)
                    G_cost = -G
                    optimizerG.step()
                else:
                    g_valid_l = autograd.Variable(Tensor(fake.size(0), 1).fill_(1.0), requires_grad=False)
                    # print (fake.size(),g_valid_l.size())
                    g_loss = adv_loss(G, g_valid_l)
                    g_loss.backward()
                    # g_loss.backward()
                    # wgan loss
                    # optimizer_G.step()
                    # G = netD(fake)
                    # G = G.mean()
                    # G.backward(mone)
                    # G_cost = -G
                    G_cost = g_loss
                optimizerG.step()

            # Write logs and save samples
            # if os.path.
            lib.plot.plot('tmp/' + DATASET + types + 'ci' + str(CRITIC_ITERS) + '/' + 'disc cost',
                          D_cost.cpu().data.numpy())
            lib.plot.plot('tmp/' + DATASET + types + 'ci' + str(CRITIC_ITERS) + '/' + 'wasserstein distance',
                          Wasserstein_D.cpu().data.numpy())
            if not FIXED_GENERATOR:
                lib.plot.plot('tmp/' + DATASET + types + 'ci' + str(CRITIC_ITERS) + '/' + 'gen cost',
                              G_cost.cpu().data.numpy())
                # torch.save(model.state_dict(), PATH)
                # Load:
                # model = TheModelClass(*args, **kwargs)

        if epoch % 10 == 1:
            lib.plot.flush()
            generate_image(_data)
            lib.plot.tick()

        if epoch % 10 == 0 or epoch == 199:
            torch.save({'generator':netG.state_dict(),'critic':netD.state_dict()},'tmp/' + DATASET + types + 'ci' + str(CRITIC_ITERS) + '/model{:d}.pt'.format(epoch))

if Finetuning:
    FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
    frame_index = [200]
    checkpoint = torch.load('tmp/' + DATASET + types + 'ci' + str(CRITIC_ITERS) + '/model50.pt')
    netD.load_state_dict(checkpoint['critic'])
    checkpoint = torch.load('tmp/20gaussianslinear_128_fintunes_sn_25_10000step_SMALLgci5/model20.pt')
    netG.load_state_dict(checkpoint['generator'])
    netG.train()
    netD.train()
    _data = data.__next__()
    real_data = torch.Tensor(_data).cuda()
    One_test = netD(torch.ones_like(real_data)).mean()
    Zero_test = netD(torch.zeros_like(real_data)).mean()
    Two_test = netD(2 * torch.ones_like(real_data)).mean()

    print(One_test, Zero_test, Two_test)
    for epoch in range(1):
        for iteration in range(ITERS):
            ############################
            # (1) Update D network
            ###########################
            for iter_d in range(CRITIC_ITERS):
                _data = data.__next__()
                real_data = torch.Tensor(_data)
                if use_cuda:
                    real_data = real_data.cuda()
                real_data_v = autograd.Variable(real_data)
                for p in netD.parameters():  # reset requires_grad
                    p.requires_grad = False
                netD.eval()
                if Output == 'linear':
                    noise = torch.randn(BATCH_SIZE, 2)
                    if use_cuda:
                        noise = noise.cuda()
                    noisev = autograd.Variable(noise).requires_grad_(False)  # totally freeze netG
                    # pdb.set_trace()
                    neg_img = autograd.Variable(netG(noisev, real_data_v).data)
                    D_original = netD(neg_img)
                    neg_img.requires_grad = True
                    for _ in range(args.step):
                        noise = torch.randn_like(neg_img, device='cuda')
                        images_out = netD(neg_img)
                        images_out.sum().backward()
                        noise = neg_img.grad.norm(dim=1).unsqueeze(1).expand(256, 2) * noise.normal_(0, args.noise) * np.sqrt(
                            args.step_lr)
                        neg_img.data.add_(noise.data)
                        neg_img.data.add_(args.step_lr, neg_img.grad.data)
                        neg_img.grad.detach_()
                        neg_img.grad.zero_()

                    neg_img = neg_img.detach()
                    for p in netD.parameters():  # reset requires_grad
                        p.requires_grad = True  # they are set to False below in netG update
                    netD.zero_grad()
                    D_fake = netD(neg_img)
                    D_real = netD(real_data_v)
                    loss = D_fake - D_real + 10.0 * (D_fake**2 +  D_real**2)
                    loss = loss.mean()
                    loss.backward()
                    # print(D_fake.requires_grad,D_real.requires_grad,neg_img.requires_grad,real_data_v.requires_grad)
                    D_cost = D_fake - D_real
                    # print(D_original - D_fake)
                    # pdb.set_trace()
                    Wasserstein_D = D_real - D_fake
                    optimizerD.step()
                    One_test = netD(torch.ones_like(real_data)).mean()
                    Zero_test = netD(torch.zeros_like(real_data)).mean()
                    Two_test = netD(2 * torch.ones_like(real_data)).mean()


        _data = data.__next__()
        One_test = netD(torch.ones_like(real_data)).mean()
        Zero_test = netD(torch.zeros_like(real_data)).mean()
        Two_test = netD(2 * torch.ones_like(real_data)).mean()

        print(One_test,Zero_test,Two_test)

        if epoch % 10 == 9 or epoch == 0:
            lib.plot.flush()
            generate_image(_data)
            lib.plot.tick()
            break

