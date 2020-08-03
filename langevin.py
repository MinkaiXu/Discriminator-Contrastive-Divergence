import chainer
import chainer.functions as F

import cupy as cp

from argparser import args


def langevin(x_fake, y_fake, dis):
    for i in range(args.num_steps):
        x_fake = langevin_step(args, x_fake, y_fake, dis, steps=i, anealing=args.anealing)
    return x_fake


def anealing_fn(steps, total_steps):
    ratio = 1
    if steps / total_steps < 1/3:
        return ratio * 1
    elif steps / total_steps < 2/3:
        return ratio * 0.7
    else:
        return ratio * 0.4
    ratio = (total_steps - steps) / total_steps
    return ratio

def langevin_step(args, x_fake, y_fake, dis, steps=None, anealing=False):
    # 1 in scale is the "Factor to rescale inputs from 0-1 box"
    noise_scale = 1. * args.noise_scale
    if steps is not None and anealing:
        noise_scale = 1. * args.noise_scale * anealing_fn(steps, args.num_steps)
    x_fake = x_fake + cp.random.normal(size=x_fake.shape,
                                       loc=0.0,
                                       scale=noise_scale)

    energy_noise = dis(x_fake, y=y_fake) * args.temperature
    # x_fake.unchain_backward()
    x_grad = chainer.grad(outputs=[energy_noise], inputs=[x_fake])[0]

    lr = args.step_lr

    if args.proj_norm != 0.0:
        if args.proj_norm_type == 'li':
            x_grad = F.clip(x_grad, -args.proj_norm, args.proj_norm)
        elif args.proj_norm_type == 'l2':
            # x_grad = tf.clip_by_norm(x_grad, args.proj_norm)
            print("L2 type of projection are not supported!!!")
            assert False
        else:
            print("Other types of projection are not supported!!!")
            assert False

    # Clip gradient norm for now
    if args.hmc:
        print("HMC is not supported!!!")
        assert False
        # Step size should be tuned to get around 65% acceptance
        # def energy(x):
        #     return args.temperature * dis(x_fake, y=y_fake)
        #             # model.forward(x, weights[0], label=LABEL_SPLIT[j], reuse=True)
        # x_last = hmc(x_fake, 15., 10, energy)
    else:
        x_last = x_fake + lr * x_grad

    x_fake = x_last
    x_fake = F.clip(x_fake, -1., 1.)  # 1："Factor to rescale inputs from 0-1 box"

    x_fake.unchain_backward()

    return x_fake
