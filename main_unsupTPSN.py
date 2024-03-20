import argparse
import logging
import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import torchvision
# import torchvision.transforms as transforms

from utils.unet2d import UNet2D
from utils.loss2D import DiceLoss,detDFunc,LapLossFunc,Hausdorff,ChanVaseLoss,betty_num
from utils.packer2D import load_skinMNISTone,vec2cor2D, save_pair, plot_grid

size = [128, 128]
mu_size = size 
dir_checkpoint = 'checkpoints/'
dir_log = 'log/'
unim_log = 'output/unim/'
if not os.path.exists(dir_checkpoint):  
    os.makedirs(dir_checkpoint)
    print("The new directory is created!   " + dir_checkpoint)
if not os.path.exists(dir_log):  
    os.makedirs(dir_log)
    print("The new directory is created!   " + dir_log)
if not os.path.exists(unim_log):  
    os.makedirs(unim_log)
    print("The new directory is created!   " + unim_log)

def train_net(unet,
              device,
              epochs=5,
              batch_size=10,
              weight = 0,
              lr=0.001,
              coefD = 10,
              coefL = 0,
              epsilon=0,
              index=0,
              pathname = ''):  
    foldername = '..//DATA_Ham//'    #'D:/PROJECT/DATA/skinMNIST/selected/'#
    trainset, testset, I_prior, coor = load_skinMNISTone(foldername, i=index)
    I_prior = I_prior.repeat(batch_size,1,1,1)
    I_prior, coor = I_prior.to(device=device, dtype=torch.float32), coor.to(device=device, dtype=torch.float32)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)    
    n_train = len(trainset)
    n_test = len(testset)

    logging.info('''Starting training:
        Epochs:             {}
        Batch size:         {}
        Learning rate:      {}
        Device:             {}
        Train Size:         {}
        Test Size:          {}
        COEF OF JACOBIAN:   {}
        COEF OF LAPLACIAN:  {}
        VALUE OF EPSILON:   {}
        INDEX:              {}
    '''.format(epochs, batch_size, lr, device.type, n_train, n_test,
               coefD,coefL,epsilon,index))

    patien = int(n_train*5)*10
    optimizer = optim.RMSprop(unet.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patien,threshold=1e-2, verbose=True)
    criterion_cvMo = ChanVaseLoss(weight, device)
    criterion_dice = DiceLoss(weight, device)
    criterion_detD = detDFunc(weight, device, size, epsilon)
    criterion_Lapl = LapLossFunc(weight, device, size[0], size[1])
    evaluatio_haus = Hausdorff(weight, device,size)

    BEST_DICE = 0
    trainF = open(pathname+ '_train.csv', 'w+')
    trainF.write('Epoch,Count,LOSS E,Dice E,detD E,Lap E,LOSS B,Dice B,detD B,Lap B\n')
    trainF.flush()
    testF = open(pathname+ '_test.csv', 'w+')
    testF.write('Epoch,Count,LOSS E,Dice E,Haus E, Betty E,detD E,Lap E,LOSS B,Dice B,Haus B, Betty D,detD B,Lap B\n')
    testF.flush()
    for epoch in range(epochs):
        unet.train()
        
        running_loss = 0.0
        running_dice = 0.0
        running_detD = 0.0
        running_Lapl = 0.0
        running_haus = 0.0

        sample_cnt = 0

        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch, epochs), unit='img') as pbar:
            for count,(IM, IM_gray, MASK) in enumerate(train_loader):   
                Nbatch = IM.shape[0]
                sample_cnt += Nbatch
                IM = IM.to(device=device, dtype=torch.float32) 
                IM_gray = IM_gray.to(device=device, dtype=torch.float32) 
                MASK = MASK.to(device=device, dtype=torch.float32) 
                I0 = I_prior[0:Nbatch,:,:,:]           
            
                vector = 2*(unet(IM)-.5)
                mapping = vec2cor2D(vector, coor)           
                MASK_pred = F.grid_sample(I0, mapping, mode='bilinear', padding_mode='border', align_corners=True)
                
                loss_cvMo = 100*criterion_cvMo(MASK_pred, IM_gray)
                loss_dice = 100*criterion_dice(MASK_pred, MASK)
                loss_detD = 100*criterion_detD(mapping)
                loss_Lapl = 100*criterion_Lapl(mapping)
                loss = loss_cvMo + coefD*loss_detD + coefL*loss_Lapl
                running_loss += Nbatch*loss.item()
                running_dice += Nbatch*(100-loss_dice.item())
                running_detD += Nbatch*loss_detD.item()
                running_Lapl += Nbatch*loss_Lapl.item()
                
                pbar.set_postfix(**{'loss (B)': loss.item(),
                                    'dice (B)': 100-loss_dice.item(),
                                    'detD (B)': loss_detD.item(),
                                    'Lapl (B)': loss_Lapl.item(),
                                    'LoCV (B)': loss_cvMo.item(),
                                    'loss (E)': running_loss / sample_cnt,
                                    'dice (E)': running_dice / sample_cnt,
                                    'detD (E)': running_detD / sample_cnt,
                                    'Lapl (E)': running_Lapl / sample_cnt})

                trainF.write('{},{},{},{},{},{},{},{},{},{}\n'.format(
                    epoch, count,
                    running_loss / sample_cnt,running_dice / sample_cnt,running_detD / sample_cnt,running_Lapl / sample_cnt,
                    loss.item(), loss_dice.item(),loss_detD.item(), loss_Lapl.item()))
                trainF.flush()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(batch_size)            

                if BEST_DICE<loss_dice.item():
                    save_pair(IM.data.cpu().numpy(), MASK_pred.data.cpu().numpy(), unim_log, index)
                    plot_grid(mapping.data.cpu().numpy(), unim_log, index)

        #unet.eval()
        #running_loss = 0.0
        #running_dice = 0.0
        #running_detD = 0.0
        #running_Lapl = 0.0
        #running_haus = 0.0
        #total_bettyn = 0.0
        #with torch.no_grad():
        #    for count, (IM, IM_gray, MASK) in enumerate(test_loader):   
        #        Nbatch = IM.shape[0]  
        #        IM = IM.to(device=device, dtype=torch.float32) 
        #        IM_gray = IM_gray.to(device=device, dtype=torch.float32) 
        #        MASK = MASK.to(device=device, dtype=torch.float32) 
        #        I0 = I_prior[0:Nbatch,:,:,:]                    
                
        #        vector = 2*(unet(IM)-.5)
        #        mapping = vec2cor2D(vector, coor)           
        #        MASK_pred = F.grid_sample(I0, mapping, mode='bilinear', padding_mode='border', align_corners=True)
                
        #        loss_cvMo = 100*criterion_cvMo(MASK_pred, IM_gray)
        #        loss_dice = 100*criterion_dice(MASK_pred, MASK)
        #        loss_detD = 100*criterion_detD(mapping)
        #        loss_Lapl = 100*criterion_Lapl(mapping)
        #        eval_haus = evaluatio_haus(MASK_pred, MASK)
        #        loss = loss_cvMo + coefD*loss_detD + coefL*loss_Lapl
        #        running_loss += Nbatch*loss.item()
        #        running_dice += Nbatch*(100-loss_dice.item())
        #        running_detD += Nbatch*loss_detD.item()
        #        running_Lapl += Nbatch*loss_Lapl.item()
        #        running_haus += Nbatch*eval_haus.item()
        #        total_bettyn += Nbatch*betty_num(MASK_pred.data.cpu().numpy())

        #if BEST_DICE > (running_dice/n_test):
        #    BEST_DICE = (running_dice/n_test)
        #    torch.save(unet.state_dict(),dir_checkpoint + 'CP_epoch{}_ce.pth'.format(epoch))
        #    logging.info('Checkpoint {} saved !'.format(epoch))   
        #print('[Epoch-%d]     Loss: %.8f     Dice: %.8f     Haus: %.8f     Betty: %.8f     detD: %.8f     Lap: %.8f' % 
        #      (epoch, 
        #       running_loss/n_test, running_dice/n_test, running_haus/n_test,total_bettyn/n_test,
        #       running_detD/n_test, running_Lapl/n_test))
        #testF.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
        #    epoch, 0,
        #    running_loss / n_test,running_dice / n_test,running_haus / n_test,total_bettyn/n_test,
        #    running_detD / n_test,running_Lapl / n_test,            0,0,0,0,0,0))
        #testF.flush()
    trainF.close()     
    testF.close()

    


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=2000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-un', '--unet', dest='unload', type=str, default=False,
                        help='Load model from a unet.pth file')    
    parser.add_argument('-w', '--weight', dest='weight', type=float, default=0,
                        help='The weight of the custom loss')
    parser.add_argument('-l1', '--lambda1', dest='lambda1', type=float, default=1,
                        help='The weight of the Jacobian loss')
    parser.add_argument('-l2', '--lambda2', dest='lambda2', type=float, default=0.01,
                        help='The weight of the Laplacian loss')
    parser.add_argument('-eps', '--epsilon', dest='epsilon', type=float, default=0,
                        help='The epsilon parameter of the ReluJacobian')
    parser.add_argument('-i', '--index', dest='index', type=int, default=2,
                        help='The index to generate for unsupervised segmentation on HAM10000')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    logging.info('Using device {}'.format(device))
    unet = UNet2D(size, device, n_channels = 3, n_classes = 2)
    if torch.cuda.device_count()>=1:
        print("Let's use ", torch.cuda.device_count(), "GPUs!")
        unet = nn.DataParallel(unet)
        args.batchsize = 1 * torch.cuda.device_count()
    if args.unload:
        unet.load_state_dict(
            torch.load(args.unload, map_location=device)
        )
        logging.info('Mapping Estor loaded from {}'.format(args.unload))
    unet.to(device=device)

    pathname= dir_log + 'unsupervisedTPSN'
    try:
        train_net(unet=unet,
                  device=device,
                  epochs=args.epochs,
                  weight = args.weight,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  coefD = args.lambda1,
                  coefL = args.lambda2,
                  epsilon = args.epsilon,
                  index=args.index,
                  pathname = pathname)
    except KeyboardInterrupt:
        torch.save(lnet.state_dict(), 'INTERRUPTED_ce.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

