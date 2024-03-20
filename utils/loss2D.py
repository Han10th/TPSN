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
        assert predict.size() == target.size(), "the size of predict and target must be equal. PRE:{} TAR:{}".format(predict.size(), target.size())
        #num = predict.size(0)        
        dims = (1, 2, 3)
        intersection = torch.sum(predict * target, dims)
        cardinality = torch.sum(predict + target, dims)
        dice_score = 2. * intersection / (cardinality + self.epsilon)
        return torch.mean(1. - dice_score)

class detDFunc(nn.Module):
    def __init__(self, weight, device, size, epsilon):
        super(detDFunc, self).__init__()
        self.weight = weight
        self.detD_loss = detD2d(device, int(size[0]), int(size[1]), epsilon)
    def forward(self, mapping):
        # pred = convert_map(mapping)
        return self.detD_loss(mapping)
def detD2d(device, H, W, epsilon):
    h, w = H-1, W-1
    hstep, wstep = 2/h, 2/w
    step = torch.tensor([wstep, hstep]).to(device=device)
    relu = torch.nn.ReLU()
    eps = epsilon
    def detD_loss(mapping):
        """
        Inputs:
            mapping: (N, 2, h, w), torch tensor
        Outputs:
            loss: (N, (h-1)*(w-1)*2), torch tensor
        """
        # N, H, W, C = mapping.shape
        mappingOO = mapping[:,0:h,0:w,:]
        mappingPO = mapping[:,0:h,1:w+1,:]
        mappingOP = mapping[:,1:h+1,0:w,:]     
        F_PO = (mappingPO - mappingOO)/step[0]
        F_OP = (mappingOP - mappingOO)/step[1]   
        det = F_PO[:,:,:,0]*F_OP[:,:,:,1] - F_PO[:,:,:,1]*F_OP[:,:,:,0]
        loss = torch.mean(relu(eps-det))
        return loss #mu
    return detD_loss

class LapLossFunc(nn.Module):
    def __init__(self, weight, device, H, W):
        super(LapLossFunc, self).__init__()
        self.lap = Laplacian(device, H, W)    
        self.weight = weight        
    def forward(self, mapping):
        lap = self.lap(mapping)
        return torch.mean(lap**2)
class Laplacian(nn.Module):
    def __init__(self, device, H, W):
        super(Laplacian, self).__init__()
        kernel = np.array([[0., 1., 0.],
                           [1., -4., 1.],
                           [0., 1., 0.]])
        h, w = H-1, W-1
        self.hstep, self.wstep = 1/h, 1/w
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel.to(device = device), requires_grad=False)
        #self.rep = nn.ReplicationPad2d(1)
 
    def forward(self, x):
        #x should be the coordinates not vector field.
        # INPUT (N,D,H,W,3) OUTPUT (N,3,D-2,H-2,W-2)
        x1 = x[:,:,:,0]
        x2 = x[:,:,:,1]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=0)/self.hstep
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=0)/self.hstep
        x = torch.cat([x1, x2], dim=1)
        return x

class Hausdorff(nn.Module):
    def __init__(self, weight, device, size):
        super(Hausdorff, self).__init__()
        self.weight = weight
        self.hausdorff_distance = HD_compute(device, float(size[0]), float(size[1]))
    def forward(self, predict, target):
        return self.hausdorff_distance(predict, target)
def HD_compute(device, H, W):
    step = torch.tensor([H, W]).to(device=device,dtype=torch.float32)
    def hausdorff_distance2d(predict, target):
        imA=torch.round(predict).to(bool)
        imB=torch.round(target).to(bool)
        N = predict.shape[0]    
        Haus = torch.zeros(N,device=device)
        for i in range(N):
            (xA1,yA1)=torch.where(imA[i,0,:,:] & ~imB[i,0,:,:])
            (xB1,yB1)=torch.where(imB[i,0,:,:])
            if xA1.shape[0] == 0:
                distA = 0
            else:
                A = torch.cat((xA1.unsqueeze(1),yA1.unsqueeze(1)),dim=1).to(torch.float32)/step
                B = torch.cat((xB1.unsqueeze(1),yB1.unsqueeze(1)),dim=1).to(torch.float32)/step
                A = A.unsqueeze(1).repeat(1,xB1.shape[0],1)
                B = B.unsqueeze(0).repeat(xA1.shape[0],1,1)
                dist_mat = torch.norm(A-B,dim=2)
                distA = torch.max(torch.min(dist_mat,dim=1)[0],dim=0)[0]
            
            (xA1,yA1)=torch.where(~imA[i,0,:,:] & imB[i,0,:,:])
            (xB1,yB1)=torch.where(imA[i,0,:,:])
            if xA1.shape[0] == 0:
                distB = 0
            else:
                A = torch.cat((xA1.unsqueeze(1),yA1.unsqueeze(1)),dim=1).to(torch.float32)/step
                B = torch.cat((xB1.unsqueeze(1),yB1.unsqueeze(1)),dim=1).to(torch.float32)/step
                A = A.unsqueeze(1).repeat(1,xB1.shape[0],1)
                B = B.unsqueeze(0).repeat(xA1.shape[0],1,1)
                dist_mat = torch.norm(A-B,dim=2)
                distB = torch.max(torch.min(dist_mat,dim=1)[0],dim=0)[0]
        
            Haus[i] = distA if distA > distB else distB
        return torch.mean(Haus)
    return hausdorff_distance2d

def betty_num(mask_batch):
    mask_batch = np.rint(mask_batch).astype(int)
    N = mask_batch.shape[0]
    total_betty = 0.0
    for i in range(N):
        mask = mask_batch[i,0,:,:]
        _,betty_no = label(mask,return_num=True)
        total_betty = total_betty+betty_no
    return total_betty/N

class ChanVaseLoss(nn.Module):
    def __init__(self, weight, device):
        super(ChanVaseLoss, self).__init__()
        self.weight = weight
        self.dims = (1, 2, 3)
    def forward(self, MASK_pred, IM_gray):
        J1 = (MASK_pred)
        J2 = (1-(MASK_pred))
        c1 = torch.sum(MASK_pred*IM_gray, self.dims, True)/torch.sum(MASK_pred, self.dims, True)
        c2 = torch.sum((1-(MASK_pred))*IM_gray, self.dims, True)/torch.sum((1-(MASK_pred)), self.dims, True)
        LossB = torch.abs(c1*J1-IM_gray) + torch.abs(c2*J2-IM_gray)
        return torch.mean(LossB)


