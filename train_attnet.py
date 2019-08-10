import sys
import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo

import datasets
import utils

from model_resnet import ResidualNet
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument(
        '--gpu',
        dest='gpu_id',
        help='GPU device id to use [0]',
        default=0,
        type=int)
    parser.add_argument(
        '--num_epochs',
        dest='num_epochs',
        help='Maximum number of training epochs.',
        default=5,
        type=int)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Batch size.',
        default=16, type=int)
    parser.add_argument(
        '--lr',
        dest='lr',
        help='Base learning rate.',
        default=0.001, type=float)
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Dataset type.',
        default='Pose_300W_LP',
        type=str)
    parser.add_argument(
        '--data_dir',
        dest='data_dir',
        help='Directory path for data.',
        default='',
        type=str)
    parser.add_argument(
        '--filename_list',
        dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='',
        type=str)
    parser.add_argument(
        '--output_string',
        dest='output_string',
        help='String appended to output snapshots.',
        default='',
        type=str)
    parser.add_argument(
        '--alpha',
        dest='alpha',
        help='Regression loss coefficient.',
        default=0.001,
        type=float)
    parser.add_argument(
        '--snapshot',
        dest='snapshot',
        help='Path of model snapshot.',
        default='',
        type=str)

    args = parser.parse_args()
    return args


def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def compute_loss(args, axis, labels, cont_labels, preds, cls_criterion,
                 reg_criterion, idx_tensor, gpu):

    if axis == "yaw":
        dim = 0
    elif axis == "pitch":
        dim = 1
    elif axis == "roll":
        dim = 2
    else:
        raise IndexError("{} is not in ['yaw', 'pitch', 'roll']".format(axis))

    label = Variable(labels[:, dim]).cuda(gpu)
    label_cont = Variable(cont_labels[:, dim]).cuda(gpu)

    loss_cls = criterion(preds, label)
    predicted = softmax(preds)
    predicted = torch.sum(predicted * idx_tensor, 1) * 3 - 99
    loss_reg = reg_criterion(predicted, label_cont)
    loss = loss_cls + alpha * loss_reg

    return loss


def compute_error(axis, cont_labels, preds, idx_tensor):

    if axis == "yaw":
        dim = 0
    elif axis == "pitch":
        dim = 1
    elif axis == "roll":
        dim = 2
    else:
        raise IndexError("{} is not in ['yaw', 'pitch', 'roll']".format(axis))

    label_cont = cont_labels[:, dim].float()
    predictions = utils.softmax_temperature(preds.data, 1)
    predictions = torch.sum(predictions * idx_tensor, 1).cpu() * 3 - 99
    error = torch.sum(torch.abs(predictions - label_cont))

    return error


def train(args, train_loader, model, criterion,
          reg_criterion, idx_tensor, optimizer,
          epoch, num_epochs, batch_num):

    for i, (images, labels, cont_labels, name) in enumerate(train_loader):
        images = Variable(images).cuda(gpu)

        # Forward pass
        yaw, pitch, roll = model(images)

        # losses
        loss_yaw = compute_loss(
            args, "yaw", labels, cont_labels, yaw, criterion,
            reg_criterion, idx_tensor, gpu)
        loss_pitch = compute_loss(
            args, "pitch", labels, cont_labels, pitch, criterion,
            reg_criterion, idx_tensor, gpu)
        loss_roll = compute_loss(
            args, "roll", labels, cont_labels, roll, criterion,
            reg_criterion, idx_tensor, gpu)

        loss_seq = [loss_yaw, loss_pitch, loss_roll]
        grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in
                    range(len(loss_seq))]
        optimizer.zero_grad()
        torch.autograd.backward(loss_seq, grad_seq)
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(('Epoch [{:d}/{:d}] Iter [{:d}/{:d}] Losses:' +
                   'Yaw {:4f}, Pitch {:4f}, Roll {:4f}').format(
                      epoch + 1, num_epochs,
                      i + 1, batch_num,
                      loss_yaw.data[0], loss_pitch.data[0], loss_roll.data[0]))


def valid(valid_loader, model, idx_tensor):

    model.eval()
    total, yaw_error, pitch_error, roll_error = 0, 0.0, 0.0, 0.0

    for i, (images, labels, cont_labels, name) in enumerate(valid_loader):

        with torch.no_grad():
            images = Variable(images).cuda(gpu)
            total += cont_labels.size(0)
            # Forward pass
            yaw, pitch, roll = model(images)
            yaw_error += compute_error(
                "yaw", cont_labels, yaw, idx_tensor)
            pitch_error += compute_error(
                "pitch", cont_labels, pitch, idx_tensor)
            roll_error += compute_error(
                "roll", cont_labels, roll, idx_tensor)

    print('Valid error in degrees ' +
          str(total) +
          ' test images. Yaw: {:4f}, Pitch: {:4f}, Roll: {:4f}'.format(
              yaw_error / total, pitch_error / total, roll_error / total))


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    # net structure
    model = ResidualNet("ImageNet", 50, 66, "CBAM")

    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose(
        [transforms.Scale(240),
         transforms.RandomCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'Synhead':
        pose_dataset = datasets.Synhead(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(
            args.data_dir,
            args.filename_list,
            transformations)

    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(
            args.data_dir,
            args.filename_list,
            transformations)

    else:
        print('Error: not a valid dataset name')
        sys.exit()

    train_size = int(0.9 * len(pose_dataset))
    valid_size = len(pose_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        pose_dataset, [train_size, valid_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2)

    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    # reg_criterion = nn.SmoothL1Loss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax(dim=1).cuda(gpu)
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    optimizer = torch.optim.Adam(
        [{'params': get_ignored_params(model), 'lr': 0},
         {'params': get_non_ignored_params(model), 'lr': args.lr},
         {'params': get_fc_params(model), 'lr': args.lr * 5}],
        lr=args.lr)

    scheduler = MultiStepLR(optimizer, milestones=[8, 18], gamma=0.1)

    print('Ready to train network.')

    for epoch in range(num_epochs):

        train(args, train_loader, model, criterion, reg_criterion,
              idx_tensor, optimizer, scheduler, epoch, num_epochs,
              len(train_dataset) // batch_size)
        valid(valid_loader, model, idx_tensor)
        scheduler.step()

        print('Saving checkpoint...')
        torch.save(
            model.state_dict(),
            'output/snapshots/' + args.output_string +
            '_epoch_' + str(epoch+1) + '.pkl')
