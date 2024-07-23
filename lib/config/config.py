import os

backbone_path = './backboneweight/hrnetv2_w48_imagenet_pretrained.pth'

datasets_root = "./datasets/"

cod_training_root = os.path.join('./datasets/DUTS-TR/')

pascal_path = os.path.join('./datasets/PASCAL-S/imgs')
ecssd_path = os.path.join( './datasets/ECSSD/imgs')
hku_path = os.path.join('./datasets/HKU-IS/imgs')
dut_omron_path = os.path.join('./datasets/DUT-O/imgs')
dut_te_path = os.path.join( './datasets/DUTS-TE/imgs')
sod_path = os.path.join('./datasets/SOD/imgs')
