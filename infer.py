import time
import datetime

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean
import cv2
from tqdm import tqdm
import skimage.io as io
import torch.nn.functional as F

from lib.config.config import *
from lib.config.misc import *
from DSINet import DSINet
from lib.py_sod_metrics import Smeasure, Emeasure, WeightedFmeasure, MAE, Fmeasure
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.manual_seed(2021)

exp_name = 'DSINet'
results_path = './predresult/'+exp_name
check_mkdir(results_path)
record_file = os.path.join(results_path, exp_name+'_metric.txt')
infertime_file = os.path.join(results_path, exp_name+'_infertime.txt')
args = {
    'scale': 384,
    'save_results': True
}
print(torch.__version__)
img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
    ('ECSSD', ecssd_path),
    ('PASCAL-S', pascal_path),
    ('DUTS-TE', dut_te_path),
    ('HKU-IS', hku_path),
    ('DUT-O', dut_omron_path)
])

results = OrderedDict()

def main_infer(model_dir):
    net = DSINet().cuda()
    net.load_state_dict(torch.load(model_dir))
    print('Load {} succeed!'.format(model_dir))

    net.eval()
    fi = open(infertime_file, 'a')
    fi.write('\n' + str(datetime.datetime.now()) + '\n')
    fi.write(('{}'.format(exp_name +'-'+ model_dir.split('/')[-1].split('.')[-2])) + '\n')
    with torch.no_grad():
        start = time.time()
        for name, root in to_test.items():
            print(root)
            time_list = []
            image_path = os.path.join(root)

            if args['save_results']:
                check_mkdir(os.path.join(results_path, model_dir.split('/')[-1].split('.')[-2], name))

            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg') or f.endswith('png')]
            for idx, img_name in enumerate(img_list):
                if name=='HKU-IS':
                    img = Image.open(os.path.join(image_path, img_name + '.png')).convert('RGB')
                else:
                    img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')

                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()

                start_each = time.time()
                prediction = net(img_var)

                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction = F.interpolate(prediction[0], (h, w), mode='bilinear', align_corners=False)
                prediction = torch.sigmoid(prediction).squeeze()  # 压缩为一个维度
                prediction = (prediction * 255.).detach().cpu().numpy().astype('uint8')

                if args['save_results']:
                    sal_path = os.path.join(results_path, model_dir.split('/')[-1].split('.')[-2], name,
                                            img_name + '.png')
                    io.imsave(sal_path, prediction)

            print(('{}'.format(exp_name+'-'+model_dir.split('/')[-1].split('.')[-2])))
            print("{}'s average Time is : {:.4f} s".format(name, mean(time_list)))
            print("{}'s average FPS is : {:.4f} fps".format(name, 1 / mean(time_list)))
            fi.write("{}'s average Time is : {:.4f} s".format(name, mean(time_list)) + '\n')
            fi.write("{}'s average FPS is : {:.4f} fps".format(name, 1 / mean(time_list))+'\n')

    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))
    fi.write("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start))))+'\n')
    fi.close()

    # start eval metrics
    file = open(record_file, 'a')
    file.write('\n' + str(datetime.datetime.now()) + '\n')
    file.write("pth: %s" % model_dir + '\n')
    for dataset in [
    'ECSSD',
    'PASCAL-S',
    'DUTS-TE',
    'HKU-IS',
    'DUT-O'
    ]:
        gt_path = "./datasets/"+dataset+"/gt/"
        predict_path=os.path.join(results_path, model_dir.split('/')[-1].split('.')[-2], dataset)

        mae = MAE()
        wfm = WeightedFmeasure()
        sm = Smeasure()
        em = Emeasure()
        fm = Fmeasure()

        images = os.listdir(predict_path)

        for image in tqdm(images):
            gt = cv2.imread(os.path.join(gt_path, image), 0)
            predict = cv2.imread(os.path.join(predict_path, image), 0)
            h, w = gt.shape
            predict = cv2.resize(predict, (w, h))
            mae.step(predict, gt)
            wfm.step(predict, gt)
            sm.step(predict, gt)
            em.step(predict, gt)
            fm.step(predict, gt)

        print(dataset)
        print('wfm: %.4f' % wfm.get_results()['wfm'])
        print('em: %.4f' % em.get_results()['em']['curve'].max())
        print('sm: %.4f' % sm.get_results()['sm'])
        print('mae: %.4f' % mae.get_results()['mae'])
        print('maxfm: %.4f' % fm.get_results()['fm']['curve'].max())

        file.write("Dataset:{:>8}".format(dataset))
        file.write(' wfm:%.4f' % wfm.get_results()['wfm'])
        file.write("  em:%.4f" % em.get_results()['em']['curve'].max())
        file.write('  sm:%.4f' % sm.get_results()['sm'])
        file.write("  mae:%.4f" % mae.get_results()['mae'])
        file.write("  maxfm:%.4f" % fm.get_results()['fm']['curve'].max() + '\n')
    file.close()


if __name__ == '__main__':
    model_dir = './ckpt/DSINet/35.pth'
    main_infer(model_dir=model_dir)
