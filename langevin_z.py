import chainer
import chainer.functions as F
import cupy as cp
from argparser import args
import math
from source.miscs.random_samples import sample_continuous
import pdb


def langevin(batchsize, gen, dis, y_fake, eval=False, given_z=None):
    if eval:
        Step_lr = args.eval_step_lr
        num_steps = args.eval_num_steps
        Noise_scale = args.eval_noise_scale
    else:
        Step_lr = args.step_lr
        num_steps = args.num_steps
        Noise_scale = args.noise_scale
    if given_z is None:
        z = sample_continuous(gen.dim_z, batchsize, distribution=gen.distribution, xp=gen.xp)
        z = chainer.Variable(z)
    else:
        z = given_z
    x_fake = gen(batchsize, z=z, y=y_fake)
    for step in range(num_steps):
        energy = dis(x_fake, y=y_fake) * args.temperature
        z_grad = chainer.grad(outputs=[energy], inputs=[z])[0]
        # pdb.set_trace()
        if args.anealing:
            step_lr = Step_lr * 0.1**(step//(num_steps/5))
            noise_scale = Noise_scale * 0.1**(step//(num_steps/5))
        else:
            step_lr = Step_lr
            noise_scale = Noise_scale
        z_grad_noise = step_lr/2*z_grad + \
            (step_lr**0.5)*cp.random.normal(size=z.shape, loc=0.0, scale=noise_scale)
        z = z + z_grad_noise
        z.unchain_backward()
        x_fake = gen(batchsize, z=z, y=y_fake)
    return x_fake, z