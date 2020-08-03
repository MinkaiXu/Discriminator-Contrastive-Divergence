import yaml

import chainer
import source.yaml_utils as yaml_utils

from argparser import args

config = yaml_utils.Config(yaml.load(open(args.config_path)))
chainer.cuda.get_device_from_id(args.gpu).use()
gen_conf = config.models['generator']
gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
dis_conf = config.models['discriminator']
dis = yaml_utils.load_model(dis_conf['fn'], dis_conf['name'], dis_conf['args'])

gen.to_gpu(device=args.gpu)
if args.G_snapshot:
    print("Load G_snapshot:{}".format(args.G_snapshot))
    chainer.serializers.load_npz(args.G_snapshot, gen)
dis.to_gpu(device=args.gpu)
if args.D_snapshot:
    print("Load D_snapshot:{}".format(args.D_snapshot))
    chainer.serializers.load_npz(args.D_snapshot, dis)
