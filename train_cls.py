"""
Author: Benny
Date: Nov 2019
"""
from dataset import ModelNetDataLoader, ScanObjectNN
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
from torch.utils.tensorboard import SummaryWriter


import argparse

from models.Menghao.model import MenghaoPointTransformerCls

import tensorboard

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser('a')
parser.add_argument('--num-points', type=int, default=1024)
parser.add_argument('-learning-rate', '--learning-rate', type=float, default=0.0001)
parser.add_argument('-weight-decay', '--weight-decay', type=float, default=0)
parser.add_argument('-batch-size', '--batch-size', type=int, default=16)
parser.add_argument('-num-class', '--num-class', type=int, default=32)
parser.add_argument('-epoch', '--epoch', type=int, default=80)
parser.add_argument('-optimizer', '--optimizer', type=str, default='Adam')
parser.add_argument('-normal', '--normal', type=bool, default=True)
#parser.add_argument('-radii', '--radii', type=int, default=1)
parser.add_argument('-radii', '--radii', nargs='+', default=[.1])

parser.add_argument('--num-points-attn', type=int, default=256)
parser.add_argument('-use-isab', '--use-isab', type=int, default=0)
parser.add_argument('-distance-function', '--distance-function', type=str, default="square")
parser.add_argument('-local-features', '--local-features', type=str, default="diff")

parser.add_argument('-dataset', '--dataset', type=str, default="modelnet")

parser.add_argument('-sampling-method', '--sampling-method', type=str, default='fps', choices=['fps','random', 'voxel'])
parser.add_argument('-downsample-layer-count', '--downsample-layer-count', type=int, default=2)
parser.add_argument('-voxel-grid-config', '--voxel-grid-config', type=int, default=0)


parser.add_argument('-model-name', '--model-name', type=str, default="Menghao")
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='GPU device number')

parser.add_argument('-exp-name', '--exp-name', type=str, required=True)

class ParserArgs:
    sampling_method: str
    downsample_layer_count: int
    use_isab: int
    num_points_attn: int
    voxel_grid_config: int


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    for j, data in (enumerate(loader)):

        points, target = data
        target = target[:, 0]
        points, target = points.to(device), target.to(device)
        classifier = model.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args: ParserArgs):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)
    
    exp_name = args.dataset + "_" + args.model_name + "_" + args.exp_name
    os.makedirs(f"log/cls/{exp_name}", exist_ok=True)

    #print(args.pretty())

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = 'modelnet40_normal_resampled/'

    if args.dataset == "modelnet":
        TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_points, split='train', normal_channel=args.normal)
        # TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    elif args.dataset == "scanobjectnn":
        TRAIN_DATASET = ScanObjectNN("train", "h5_files/main_split_nobg", args.num_points)
        # TEST_DATASET = ScanObjectNN("test", "h5_files/main_split_nobg")
    # TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    train_dataset, val_dataset = torch.utils.data.random_split(TRAIN_DATASET, [0.9, 0.1], generator=torch.Generator().manual_seed(42))

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    writer = SummaryWriter(f"log/cls/{exp_name}")
    
    


    '''MODEL LOADING'''
    if args.dataset == "scanobjectnn":
        args.num_class = 15
        args.normal = False
    elif args.dataset == "modelnet":
        args.num_class = 40

    args.input_dim = 6 if args.normal else 3
    # shutil.copy('models/{}/model.py'.format(args.model_name), '.')
    import sys
    print(sys.argv[1:])

    if args.model_name == "Menghao":
        classifier = MenghaoPointTransformerCls(args).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load(f'log/cls/{exp_name}/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model')
    except:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_epoch = 0
    mean_correct = []

    '''TRANING'''
    print('Start training...')
    for epoch in range(start_epoch,args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        
        import time
        
        start = time.time()
        
        classifier.train()
        for batch_id, data in enumerate(trainDataLoader, 0):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points, target = points.to(device), target.to(device)
            optimizer.zero_grad()

            pred = classifier(points)
            loss = criterion(pred, target.long())
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            
        scheduler.step()
        
        end = time.time()

        train_instance_acc = np.mean(mean_correct)
        print('Train Instance Accuracy: %f' % train_instance_acc)
        
        # record run time.
        with open(f"log/cls/{exp_name}/times.txt", "a") as f:
            f.write(f"{epoch},{end - start}\n")
        
        writer.add_scalar("train_acc", train_instance_acc, epoch)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=args.num_class)
            
            writer.add_scalar("val_acc", instance_acc, epoch)
            writer.flush()


            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            print('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            print('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
            
            with open(f"log/cls/{exp_name}/val_acc.txt", "a") as f:
                f.write('Test Instance Accuracy: %f, Class Accuracy: %f\n'% (instance_acc, class_acc))

            if (instance_acc >= best_instance_acc):
                print('Save model...')
                savepath = f'log/cls/{exp_name}/best_model.pth'
                print('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
    
    # Log test set.
    import test_cls
    test_cls.main(args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)