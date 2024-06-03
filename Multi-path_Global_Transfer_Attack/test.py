import numpy as np
import argparse
import torch
import torch.utils.data as dataf
import torch.nn as nn
from scipy import io

from Targeted_Models.models import HeEtAl
from Targeted_Models.swintransformer import SwinTransformer
from trans_pso_attack import psodim_attack

def loadData():
    DataPath = '900(1000)_PaviaU01/paviaU.mat'
    TRPath = '900(1000)_PaviaU01/TRLabel.mat'
    TSPath = '900(1000)_PaviaU01/TSLabel.mat'

    # load data
    Data = io.loadmat(DataPath)
    TrLabel = io.loadmat(TRPath)
    TsLabel = io.loadmat(TSPath)

    Data = Data['paviaU']
    Data = Data.astype(np.float32)
    TrLabel = TrLabel['TRLabel']
    TsLabel = TsLabel['TSLabel']

    return Data, TrLabel, TsLabel


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def createPatches(X, y, windowSize):
    [m, n, l] = np.shape(X)
    temp = X[:, :, 0]
    pad_width = np.floor(windowSize / 2)
    pad_width = np.int_(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    x2 = np.empty((m2, n2, l), dtype='float32')

    for i in range(l):
        temp = X[:, :, i]
        pad_width = np.floor(windowSize / 2)
        pad_width = np.int_(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2[:, :, i] = temp2

    [ind1, ind2] = np.where(y != 0)
    TrainNum = len(ind1)
    patchesData = np.empty((TrainNum, l, windowSize, windowSize), dtype='float32')
    patchesLabels = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch = np.reshape(patch, (windowSize * windowSize, l))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (l, windowSize, windowSize))
        patchesData[i, :, :, :] = patch
        patchlabel = y[ind1[i], ind2[i]]
        patchesLabels[i] = patchlabel

    return patchesData, patchesLabels


def Normalize(dataset):
    [m, n, b] = np.shape(dataset)
    # change to [0,1]
    for i in range(b):
        _range = np.max(dataset[:, :, i]) - np.min(dataset[:, :, i])
        dataset[:, :, i] = (dataset[:, :, i] - np.min(dataset[:, :, i])) / _range

    # #standardization
    # mean = np.zeros(b)
    # std = np.zeros(b)
    #
    # for i in range(b):
    #     mean[i] = np.mean(dataset[:, :, i])
    #     std[i] = np.std(dataset[:, :, i])
    #     dataset[:,:,i] = (dataset[:,:,i] - mean[i]) / std[i]

    return dataset


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def testFull(args, device, model, Data, windowSize):
    # show the whole image
    # The whole data is too big to test in one time; So dividing it into several parts
    part = args.test_batch_size
    [m, n, b] = np.shape(Data)
    x = Data
    temp = x[:, :, 0]
    pad_width = np.floor(windowSize / 2)
    pad_width = np.int_(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    x2 = np.empty((m2, n2, b), dtype='float32')

    for i in range(b):
        temp = x[:, :, i]
        pad_width = np.floor(windowSize / 2)
        pad_width = np.int_(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2[:, :, i] = temp2

    pred_all = np.empty((m * n, 1), dtype='float32')
    number = m * n // part
    for i in range(number):
        D = np.empty((part, b, windowSize, windowSize), dtype='float32')
        count = 0
        for j in range(i * part, (i + 1) * part):
            row = j // n
            col = j - row * n
            row2 = row + pad_width
            col2 = col + pad_width
            patch = x2[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
            patch = np.reshape(patch, (windowSize * windowSize, b))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (b, windowSize, windowSize))
            D[count, :, :, :] = patch
            count += 1

        temp = torch.from_numpy(D)
        # temp = temp.cuda()
        temp = temp.to(device)
        temp2 = model(temp)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_all[i * part:(i + 1) * part, 0] = temp3.cpu()
        del temp, temp2, temp3, D

    if (i + 1) * part < m * n:
        D = np.empty((m * n - (i + 1) * part, b, windowSize, windowSize), dtype='float32')
        print(D.shape)
        count = 0
        for j in range((i + 1) * part, m * n):
            row = j // n
            col = j - row * n
            row2 = row + pad_width
            col2 = col + pad_width
            patch = x2[(row2 - pad_width):(row2 + pad_width), (col2 - pad_width):(col2 + pad_width), :]
            patch = np.reshape(patch, (windowSize * windowSize, b))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (b, windowSize, windowSize))
            D[count, :, :, :] = patch
            count += 1

        temp = torch.from_numpy(D)
        temp = temp.to(device)
        temp2 = model(temp)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_all[(i + 1) * part:m * n, 0] = temp3.cpu()
        del temp, temp2, temp3, D

    #pred_all = np.reshape(pred_all, (m, n)) + 1
    #io.savemat(args.save_path, {'PredAll': pred_all})


def TrainFuc(args, device, Data, TR_gt, TS_gt, windowSize):
    Data_TR, TR_gt_M = createPatches(Data, TR_gt, windowSize)
    Data_TS, TS_gt_M = createPatches(Data, TS_gt, windowSize)

    # change to the input type of PyTorch
    Data_TR = torch.from_numpy(Data_TR)
    Data_TS = torch.from_numpy(Data_TS)

    TrainLabel = torch.from_numpy(TR_gt_M) - 1
    TrainLabel = TrainLabel.long()
    TestLabel = torch.from_numpy(TS_gt_M) - 1
    TestLabel = TestLabel.long()

    return Data_TR, Data_TS, TrainLabel, TestLabel


def str2bool(v):
    return v.lower() in ('true', '1')



def Action(epsilon):
    parser = argparse.ArgumentParser(description='PyTorch')
    # 文中设置的参数
    parser.add_argument('--batch_size', type=int, default=128,
                        help='# of images in each batch of data')
    parser.add_argument('--is_train', type=str2bool, default=True,
                        help='Whether to train or test the model')
    parser.add_argument('--epochs', type=int, default=300,
                        help='# of epochs to train for')
    parser.add_argument('--init_lr', type=float, default=1e-3,
                        help='Initial learning rate value')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='value of weight dacay for regularization')
    parser.add_argument('--use_gpu', type=str2bool, default=False,
                        help="Whether to run on the GPU")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_path', default='./checkpoint/', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test-batch-size', type=int, default=400, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epsilon', type=float, default=0.01, metavar='LR',
                        help='adversarial rate (default: 0.1)')

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.epsilon = epsilon

    print("CUDA Available: ", torch.cuda.is_available())
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    Data, TR_gt, TS_gt = loadData()

    [m, n, b] = np.shape(Data)
    Classes = len(np.unique(TR_gt)) - 1

    Data = Normalize(Data)
    # ===============================================================================================
    windowSize = 32

    pretrained_model = "900(1000)_PaviaU01/Train/SwinTransformer.pkl"
    tartrained_model1 = "900(1000)_PaviaU01/Train/M3D_DCNN.pkl"

    pre_model = SwinTransformer(patch_size=32, in_chans=b, num_classes=Classes).to(device)
    pre_model.load_state_dict(torch.load(pretrained_model))
    pre_model.eval()

    targeted_model1 = HeEtAl(b, Classes, patch_size=windowSize).to(device)
    targeted_model1.load_state_dict(torch.load(tartrained_model1))
    targeted_model1.eval()


    # print('Training window size:', windowSize)

    [Data_tr, Data_ts, TrainLabel, TestLabel] = TrainFuc(args, device, Data, TR_gt, TS_gt, windowSize)
    del Data_tr, TrainLabel

    # =================================================k=====================================================================
    # Adversarial training setup
    # adversary = FGSMAttack(epsilon=0.3)
    for epoch in range(1):
        part = args.test_batch_size
        number = len(TestLabel) // part

        # Test_advSample
        num_correct_adv_pre = 0
        num_correct_adv_tar1 = 0
        for i in range(number):
            tempdata = Data_ts[i * part:(i + 1) * part, :, :]
            TestLabel_1 = TestLabel[i * part:(i + 1) * part]
            tempdata = tempdata.to(device)
            TestLabel_1 = TestLabel_1.to(device)

            x_adv = psodim_attack(pre_model, device, tempdata, TestLabel_1, epsilon)

            out_C = torch.argmax(pre_model(x_adv), 1)
            num_correct_adv_pre += torch.sum(out_C == TestLabel_1, 0)
            del out_C

            out_C = torch.argmax(targeted_model1(x_adv), 1)
            num_correct_adv_tar1 += torch.sum(out_C == TestLabel_1, 0)
            del out_C

            del tempdata, TestLabel_1, x_adv

        if (i + 1) * part < len(TestLabel):
            tempdata = Data_ts[(i + 1) * part:len(TestLabel), :, :]
            TestLabel_1 = TestLabel[(i + 1) * part:len(TestLabel)]
            tempdata = tempdata.to(device)
            TestLabel_1 = TestLabel_1.to(device)

            x_adv = psodim_attack(pre_model, device, tempdata, TestLabel_1, epsilon)

            out_C = torch.argmax(pre_model(x_adv), 1)
            num_correct_adv_pre += torch.sum(out_C == TestLabel_1, 0)
            del out_C

            out_C = torch.argmax(targeted_model1(x_adv), 1)
            num_correct_adv_tar1 += torch.sum(out_C == TestLabel_1, 0)
            del out_C


            del tempdata, TestLabel_1, x_adv

        print('epsilon =', epsilon)
        print(pretrained_model)
        print('num_correct_adv_pre: ', num_correct_adv_pre)
        print('accuracy of adv_pre test set: %f\n' % (num_correct_adv_pre.item() / len(TestLabel)))

        print(tartrained_model1)
        print('num_correct_adv_tar1: ', num_correct_adv_tar1)
        print('accuracy of adv_tar1 test set: %f\n' % (num_correct_adv_tar1.item() / len(TestLabel)))

def main():
    epsilon = [0.01, 0.03]
    for i in epsilon:
        Action(i)

if __name__ == '__main__':
    main()