import os, sys
sys.path.append(os.getcwd())
import pdb
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import tflib as lib
import tflib.save_images
import tflib.mnist
import tflib.plot
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mcmc_sampler import spectral_norm

torch.manual_seed(1)
torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 400 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
Epochs = 200

if not os.path.exists('tmp/mnist'):
    os.mkdir('tmp/mnist')
lib.print_model_settings(locals().copy())

# ==================Definition Start======================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        return output.view(-1, OUTPUT_DIM)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            spectral_norm(nn.Conv2d(1, DIM, 5, stride=2, padding=2)),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2)),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            spectral_norm(nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2)),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = spectral_norm(nn.Linear(4*4*4*DIM, 1))

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1)

def generate_image(frame, netG):
    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise).requires_grad_(False)
    samples = netG(noisev)
    samples = samples.view(BATCH_SIZE, 28, 28)
    # print samples.size()

    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(
        samples,
        'tmp/mnist1/samples_{}.png'.format(frame)
    )

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# ==================Definition End======================

netG = Generator()
netD = Discriminator()
print (netG)
print (netD)

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

data = inf_train_gen()

for epoch in range(Epochs):
    for iteration in range(ITERS):
    # for iteration in range(ITERS):
        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in range(CRITIC_ITERS):
            _data = data.__next__()
            real_data = torch.Tensor(_data)
            if use_cuda:
                real_data = real_data.cuda(gpu)
            real_data_v = autograd.Variable(real_data)

            netD.zero_grad()

            # train with real
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            # print D_real
            D_real.backward(mone)

            # train with fake
            noise = torch.randn(BATCH_SIZE, 128)
            if use_cuda:
                noise = noise.cuda(gpu)
            noisev = autograd.Variable(noise).requires_grad_(False)  # totally freeze netG# totally freeze netG
            fake = autograd.Variable(netG(noisev).data)
            inputv = fake
            # pdb.set_trace()
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)

            # train with gradient penalty
            # gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            # gradient_penalty.backward()

            D_cost = D_fake - D_real
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()

        noise = torch.randn(BATCH_SIZE, 128)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise)
        fake = netG(noisev)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        # print(G.requires_grad)
        optimizerG.step()

        # Write logs and save samples
    lib.plot.plot('tmp/mnist/time', time.time() - start_time)
    lib.plot.plot('tmp/mnist/train disc cost', D_cost.cpu().data.numpy())
    lib.plot.plot('tmp/mnist/train gen cost', G_cost.cpu().data.numpy())
    lib.plot.plot('tmp/mnist/wasserstein distance', Wasserstein_D.cpu().data.numpy())

        # Calculate dev loss and generate samples every 100 iters
    if epoch % 10 == 9:
        dev_disc_costs = []
        for images,_ in dev_gen():
            imgs = torch.Tensor(images)
            if use_cuda:
                imgs = imgs.cuda(gpu)
            imgs_v = autograd.Variable(imgs).requires_grad_(False)

            D = netD(imgs_v)
            _dev_disc_cost = -D.mean().cpu().data.numpy()
            dev_disc_costs.append(_dev_disc_cost)
        lib.plot.plot('tmp/mnist/dev disc cost', np.mean(dev_disc_costs))
        generate_image(epoch, netG)

    # Write logs every 100 iters
    if (epoch < 5) or (epoch % 10 == 9):
        lib.plot.flush()
    if epoch % 10 == 9:
        torch.save({'generator': netG.state_dict(), 'critic': netD.state_dict()},
                   'tmp/mnist/model{:d}.pt'.format(epoch))
        pdb.set_trace()

    lib.plot.tick()

if Finetuning:
    FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
    frame_index = [200]
    checkpoint = torch.load('tmp/' + DATASET + '/model50.pt')
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
                    for _ in range(60):
                        noise = torch.randn_like(neg_img, device='cuda')
                        images_out = netD(neg_img)
                        images_out.sum().backward()
                        noise = neg_img.grad.norm(dim=1).unsqueeze(1).expand(256, 2) * noise.normal_(0, 0.2) * np.sqrt(
                            0.2)
                        neg_img.data.add_(noise.data)
                        neg_img.data.add_(0.2, neg_img.grad.data)
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

                    # print(One_test, Zero_test, Two_test)

            # if not FIXED_GENERATOR and epoch < 10:
            #     ############################
            #     # (2) Update G network
            #     ###########################
            #     for p in netD.parameters():
            #         p.requires_grad = False  # to avoid computation
            #     netG.zero_grad()
            #     _data = data.__next__()
            #     real_data = torch.Tensor(_data)
            #     if use_cuda:
            #         real_data = real_data.cuda()
            #     real_data_v = autograd.Variable(real_data)
            #
            #     noise = torch.randn(BATCH_SIZE, 2)
            #     if use_cuda:
            #         noise = noise.cuda()
            #     noisev = autograd.Variable(noise)
            #     fake = netG(noisev, real_data_v)
            #     G = netD(fake)
            #     if Output == 'linear':
            #         G = netD(fake)
            #         G = G.mean()
            #         G.backward(mone)
            #         G_cost = -G
            #         optimizerG.step()
            #     # optimizerG.step()

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


