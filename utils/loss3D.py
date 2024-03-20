import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label
import numpy as np
class DiceLoss(nn.Module):
    def __init__(self, weight, device):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.epsilon = 1e-5    
    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        #num = predict.size(0)        
        dims = (2, 3, 4)
        intersection = torch.sum(predict * target, dims)
        cardinality = torch.sum(predict + target, dims)
        dice_score = 2. * intersection / (cardinality + self.epsilon)
        return torch.mean(1. - dice_score)

class detDFunc(nn.Module):
    def __init__(self, weight, device, size, epsilon):
        super(detDFunc, self).__init__()
        self.weight = weight
        self.detD_loss = detD3d(device, int(size[0]), int(size[1]), int(size[2]), epsilon)
    def forward(self, mapping):
        # pred = convert_map(mapping)
        return self.detD_loss(mapping)
def detD3d(device, D, H, W, epsilon):
    d, h, w = D-1, H-1, W-1
    dstep, hstep, wstep = 2/d, 2/h, 2/w
    step = torch.tensor([wstep, hstep, dstep]).to(device=device)
    relu = torch.nn.ReLU()
    eps = epsilon
    def detD_loss(mapping):
        """
        Inputs:
            mapping: (N, 2, h, w), torch tensor
        Outputs:
            loss: (N, (h-1)*(w-1)*2), torch tensor
        """
        # N, D, H, W, C = mapping.shape
        mappingOOO = mapping[:,0:d,0:h,0:w,:]
        mappingPOO = mapping[:,0:d,0:h,1:w+1,:]
        mappingOPO = mapping[:,0:d,1:h+1,0:w,:]
        mappingOOP = mapping[:,1:d+1,0:h,0:w,:]        
        F_POO = (mappingPOO - mappingOOO)/step[0]
        F_OPO = (mappingOPO - mappingOOO)/step[1]
        F_OOP = (mappingOOP - mappingOOO)/step[2]       
        M1 = F_POO[:,:,:,:,0] * (F_OPO[:,:,:,:,1]*F_OOP[:,:,:,:,2] - F_OOP[:,:,:,:,1]*F_OPO[:,:,:,:,2])
        M2 = F_OPO[:,:,:,:,0] * (F_POO[:,:,:,:,1]*F_OOP[:,:,:,:,2] - F_OOP[:,:,:,:,1]*F_POO[:,:,:,:,2])
        M3 = F_OOP[:,:,:,:,0] * (F_POO[:,:,:,:,1]*F_OPO[:,:,:,:,2] - F_OPO[:,:,:,:,1]*F_POO[:,:,:,:,2])
        det = M1 - M2 + M3
        loss = torch.mean(relu(eps-det))
        return loss #mu
    return detD_loss

class LapLossFunc(nn.Module):
    def __init__(self, weight, device, D, H, W):
        super(LapLossFunc, self).__init__()
        self.lap = Laplacian(device, D, H, W)
        self.weight = weight        
    def forward(self, mapping):
        lap = self.lap(mapping)
        return torch.mean(lap**2)
class Laplacian(nn.Module):
    def __init__(self, device, D, H, W):
        super(Laplacian, self).__init__()
        kernel = np.array([[[0., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 0.]],
                           
                           [[0., 1., 0.],
                            [1., -6., 1.],
                            [0., 1., 0.]],

                           [[0., 0., 0.],
                            [0., 1., 0.],
                            [0., 0., 0.]]])
        d, h, w = D-1, H-1, W-1
        self.dstep, self.hstep, self.wstep = 1/d, 1/h, 1/w
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel.to(device = device), requires_grad=False)
        #self.rep = nn.ReplicationPad2d(1)
 
    def forward(self, x):
        #x should be the coordinates not vector field.
        # INPUT (N,D,H,W,3) OUTPUT (N,3,D-2,H-2,W-2)
        x1 = x[:,:,:,:,0]
        x2 = x[:,:,:,:,1]
        x3 = x[:,:,:,:,2]
        #x1 = F.conv2d(self.rep(x1.unsqueeze(1)), self.weight, padding=0)
        #x2 = F.conv2d(self.rep(x2.unsqueeze(1)), self.weight, padding=0)
        x1 = F.conv3d(x1.unsqueeze(1), self.weight, padding=0)/self.hstep
        x2 = F.conv3d(x2.unsqueeze(1), self.weight, padding=0)/self.hstep
        x3 = F.conv3d(x3.unsqueeze(1), self.weight, padding=0)/self.hstep
        x = torch.cat([x1, x2, x3], dim=1)
        return x

class Hausdorff(nn.Module):
    def __init__(self, weight, device, size):
        super(Hausdorff, self).__init__()
        self.weight = weight
        self.hausdorff_distance = HD_compute(device, float(size[0]), float(size[1]), float(size[2]))
    def forward(self, predict, target):
        return self.hausdorff_distance(predict, target)
def HD_compute(device, D, H, W):
    step = torch.tensor([D, H, W]).to(device=device,dtype=torch.float32)
    def hausdorff_distance3d(predict, target):
        imA=torch.round(predict).to(bool)
        imB=torch.round(target).to(bool)
        N = predict.shape[0]    
        Haus = torch.zeros(N,device=device)
        for i in range(N):
            (xA1,yA1,zA1)=torch.where(imA[i,0,:,:,:] & ~imB[i,0,:,:,:])
            (xB1,yB1,zB1)=torch.where(imB[i,0,:,:,:])
            if xA1.shape[0] == 0:
                distA = 0
            else:
                A = torch.cat((xA1.unsqueeze(1),yA1.unsqueeze(1),zA1.unsqueeze(1)),dim=1).to(torch.float32)/step
                B = torch.cat((xB1.unsqueeze(1),yB1.unsqueeze(1),zB1.unsqueeze(1)),dim=1).to(torch.float32)/step
                A = A.unsqueeze(1).repeat(1,xB1.shape[0],1)
                B = B.unsqueeze(0).repeat(xA1.shape[0],1,1)
                dist_mat = torch.norm(A-B,dim=2)
                distA = torch.max(torch.min(dist_mat,dim=1)[0],dim=0)[0]
            
            (xA1,yA1,zA1)=torch.where(~imA[i,0,:,:,:] & imB[i,0,:,:,:])
            (xB1,yB1,zB1)=torch.where(imA[i,0,:,:,:])
            if xA1.shape[0] == 0:
                distB = 0
            else:
                if xB1.shape[0] == 0:
                    distB = 999
                else:
                    A = torch.cat((xA1.unsqueeze(1),yA1.unsqueeze(1),zA1.unsqueeze(1)),dim=1).to(torch.float32)/step
                    B = torch.cat((xB1.unsqueeze(1),yB1.unsqueeze(1),zB1.unsqueeze(1)),dim=1).to(torch.float32)/step
                    A = A.unsqueeze(1).repeat(1,xB1.shape[0],1)
                    B = B.unsqueeze(0).repeat(xA1.shape[0],1,1)
                    dist_mat = torch.norm(A-B,dim=2)
                    distB = torch.max(torch.min(dist_mat,dim=1)[0],dim=0)[0]
        
            Haus[i] = distA if distA > distB else distB
        return torch.mean(Haus)
    return hausdorff_distance3d

def betty_num(mask_batch):
    mask_batch = np.rint(mask_batch).astype(int)
    N = mask_batch.shape[0]
    total_betty = 0.0
    for i in range(N):
        mask = mask_batch[i,0,:,:,:]
        _,betty_no = label(mask,return_num=True)
        #betty_no = bwlabel3(mask)
        total_betty = total_betty+betty_no
    return total_betty/N