import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from lib.models.cls_hrnet import HighResolutionNet
from lib.models.swin import SwinTransformer
from lib.models.capsules import PWV
# state_dict_path = './backboneweight/hrnetv2_w48_imagenet_pretrained.pth'
# with open(
#         r"./lib/config/cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml",
#         'r', encoding='utf-8') as f:
#     temp = yaml.load(stream=f, Loader=yaml.FullLoader)
# hrnet = HighResolutionNet(temp)
# hrnet.load_state_dict(torch.load(state_dict_path))

swin_path='./backboneweight/swin_base_patch4_window12_384_22k.pth'
swin=SwinTransformer(img_size=384,embed_dim=128,depths=[2,2,18,2],num_heads=[4,8,16,32], window_size=384//32)
swin.load_state_dict(torch.load(swin_path, map_location='cpu')['model'], strict=False)


