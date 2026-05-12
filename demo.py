# -*- coding:utf-8 -*-
import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
from scipy.io import loadmat
from MDAF_Net import MDAF
import numpy as np
import time
import os
from utils import train_patch, setup_seed, output_metric, print_args, train_epoch, valid_epoch

# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("GLT")
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--test_freq', type=int, default=20, help='number of evaluation')
parser.add_argument('--epoches', type=int, default=500, help='epoch number')  # Muufl 200
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')  # diffGrad 1e-3
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston'], default='Houston', help='dataset to use')
parser.add_argument('--num_classes', choices=[11, 6, 15], default=15, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--patches1', type=int, default=8, help='number1 of patches')
parser.add_argument('--patches2', type=int, default=16, help='number2 of patches')
parser.add_argument('--patches3', type=int, default=24, help='number3 of patches')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'],
                    default='one_time', help='training times')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

All_OA = []
All_AA = []
All_Kappa = []
All_CA = []


def train_1times(number):
    # setup_seed(args.seed)
    # -------------------------------------------------------------------------------
    # prepare data
    # setup_seed(args.seed)
    if args.dataset == 'Houston':
        num_classes = 15
        DataPath1 = './dataset/Houston/Houston.mat'
        DataPath2 = './dataset/Houston/LiDAR.mat'
        LabelPath_10TIMES = './dataset/Houston/20/train_test_gt_{}.mat'.format(number)
        Data1 = loadmat(DataPath1)['img']
        Data2 = loadmat(DataPath2)['img']
    elif args.dataset == 'Muufl':
        num_classes = 11
        DataPath1 = './dataset/Muufl/hsi.mat'
        DataPath2 = './dataset/Muufl/lidar_DEM.mat'
        Data1 = loadmat(DataPath1)['hsi']
        Data2 = loadmat(DataPath2)['lidar']
        LabelPath_10TIMES = './dataset/Muufl/20/train_test_gt_{}.mat'.format(number)
    elif args.dataset == 'Trento':
        num_classes = 11
        DataPath1 = './dataset/Trento/HSI.mat'
        DataPath2 = './dataset/Trento/LiDAR.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['LiDAR']
        LabelPath_10TIMES = './dataset/Trento/20/train_test_gt_{}.mat'.format(number)
    TrLabel_10TIMES = loadmat(LabelPath_10TIMES)['train_data']  # 349*1905
    TsLabel_10TIMES = loadmat(LabelPath_10TIMES)['test_data']  # 349*1905
    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)
    patchsize1 = args.patches1  # input spatial size for 2D-CNN
    pad_width1 = np.floor(patchsize1 / 2)
    pad_width1 = int(pad_width1)  # 8
    patchsize2 = args.patches2  # input spatial size for 2D-CNN
    pad_width2 = np.floor(patchsize2 / 2)
    pad_width2 = int(pad_width2)  # 8
    patchsize3 = args.patches3  # input spatial size for 2D-CNN
    pad_width3 = np.floor(patchsize3 / 2)
    pad_width3 = int(pad_width3)  # 8
    TrainPatch11, TrainPatch21, TrainLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TrLabel_10TIMES)
    TestPatch11, TestPatch21, TestLabel = train_patch(Data1, Data2, patchsize1, pad_width1, TsLabel_10TIMES)
    TrainPatch12, TrainPatch22, _ = train_patch(Data1, Data2, patchsize2, pad_width2, TrLabel_10TIMES)
    TestPatch12, TestPatch22, _ = train_patch(Data1, Data2, patchsize2, pad_width2, TsLabel_10TIMES)
    TrainPatch13, TrainPatch23, _ = train_patch(Data1, Data2, patchsize3, pad_width3, TrLabel_10TIMES)
    TestPatch13, TestPatch23, _ = train_patch(Data1, Data2, patchsize3, pad_width3, TsLabel_10TIMES)
    train_dataset = Data.TensorDataset(TrainPatch11, TrainPatch21, TrainPatch12, TrainPatch22, TrainPatch13,
                                       TrainPatch23, TrainLabel)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = Data.TensorDataset(TestPatch11, TestPatch21, TestPatch12, TestPatch22, TestPatch13, TestPatch23,
                                      TestLabel)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
    print("height2={0},width2={1},band2={2}".format(height2, width2, band2))
    # -------------------------------------------------------------------------------
    # create model
    model = MDAF(l1=band1, l2=band2, patch_size=args.patches1, num_patches=64, num_classes=args.num_classes,encoder_embed_dim=64, decoder_embed_dim=32)
    model = model.cuda()
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
    # -------------------------------------------------------------------------------
    # train and test
    if args.flag_test == 'train':
        BestAcc = 0
        val_acc = []
        print("start training")
        tic = time.time()
        for epoch in range(args.epoches):
            # train model
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, train_loader, criterion, optimizer)
            OA1, AA1, Kappa1, CA1 = output_metric(tar_t, pre_t)
            print("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f} | train_AA: {:.4f} | train_Kappa: {:.4f}"
                  .format(epoch + 1, train_obj, OA1, AA1, Kappa1))
            scheduler.step()

            if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, test_loader, criterion)
                OA2, AA2, Kappa2, CA2 = output_metric(tar_v, pre_v)
                val_acc.append(OA2)
                print("Every 5 epochs' records:")
                print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA2, Kappa2))
                print(CA2)
                if OA2 > BestAcc:
                    torch.save(model.state_dict(), './GLT_Net.pkl')
                    BestAcc = OA2

        toc = time.time()
        model.eval()
        model.load_state_dict(torch.load('./GLT_Net.pkl'))
        tar_v, pre_v = valid_epoch(model, test_loader, criterion)
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)

        All_OA.append(OA)
        All_AA.append(AA)
        All_Kappa.append(Kappa)
        All_CA.append(CA)
        print("Final records:")
        print("Maxmial Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
        print(CA)
        print("Running Time: {:.2f}".format(toc - tic))
        print("**************************************************")
        print("Parameter:")
        print_args(vars(args))


if __name__ == '__main__':
    setup_seed(args.seed)
    tic = time.time()
    for i in range(1, 11):
        print('-----------------this is the {}----------------------'.format(i))
        train_1times(i)
    toc = time.time()
    if All_OA:
        avg_OA = np.mean(All_OA) * 100
        avg_AA = np.mean(All_AA) * 100
        avg_Kappa = np.mean(All_Kappa)

        # 计算所有CA的平均值
        avg_CA = np.mean(All_CA, axis=0)

        print('-----------average-----------')
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(avg_OA, avg_AA, avg_Kappa))
        print(avg_CA)
        print("Running Time: {:.2f}s".format((toc - tic) / 10))
    else:
        print("No valid results to calculate averages.")



