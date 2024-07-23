import datetime
import time
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm

from lib import joint_transforms, loss
from lib.config.config import cod_training_root
from datasets.datasets import ImageFolder
from lib.config.misc import AvgMeter, check_mkdir
from DSINet import DSINet
from infer import main_infer


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
cudnn.benchmark = True

torch.manual_seed(2021)

ckpt_path = './ckpt'
exp_name = 'DSINet'

args = {
    'epoch_num': 35,
    'train_batch_size': 7,
    'last_epoch': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 384,
    'save_point': 5,
    'poly_train': True,
    'optimizer': 'SGD',
}

print(torch.__version__)

write = exp_name + '_log'
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, write + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Transform Data.
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# Prepare Data Set.
train_set = ImageFolder(cod_training_root, joint_transform, img_transform, target_transform)
print("Train set: {} images".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True)
total_epoch = args['epoch_num'] * len(train_loader)

# loss function
structure_loss = loss.structure_loss().cuda()

def main():
    print(args)
    print(exp_name)
    net = DSINet().cuda().train()
    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    open(log_path, 'w').write(exp_name + '\n' + str(datetime.datetime.now()) + '\n'+str(args) + '\n\n')
    train(net, optimizer)
    writer.close()

def train(net, optimizer):
    curr_iter = 1
    start_time = time.time()

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):

        model_dir=os.path.join(ckpt_path,exp_name,str(epoch) + '.pth')
        loss_record, loss_1_record = AvgMeter(), AvgMeter()

        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            predict = net(inputs)  #[out,d1,d2,d3]

            loss_1 = structure_loss(predict[0], labels)
            loss_2 = structure_loss(predict[1], labels)
            loss_3 = structure_loss(predict[2], labels)
            loss_4 = structure_loss(predict[3], labels)
            loss = loss_1 + loss_2 * 0.8 + loss_3 * 0.8 + loss_4 * 0.8

            loss.backward()
            optimizer.step()
            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)

            if curr_iter % 10 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)

            log = '[epoch%3d,cur_iter%6d,baselr%.6f,loss=%.5f,loss1=%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_1_record.avg)

            train_iterator.set_description(log)
            if curr_iter % 100 ==0 or curr_iter%len(train_loader)==0:
                open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        if epoch % args['save_point'] == 0:
            net.cpu()
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            main_infer(model_dir=model_dir)
            net.cuda()

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            open(log_path, 'a').write(str(datetime.datetime.now()) + '\n'+"Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name,"Optimization Have Done!")
            return


if __name__ == '__main__':
    main()

