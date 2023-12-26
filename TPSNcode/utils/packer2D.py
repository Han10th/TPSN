import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat,savemat
from skimage.measure import label
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from torch.utils.data import TensorDataset,Dataset
from torchvision.io import read_image

def make_prior(foldername = '../scratch/data_kits/',size=[512,512]):
    mask_total = np.zeros(size)
    for phase in ['train','test']:
        maskfolderpath = foldername + phase + '/mask/'
        mask_list = os.listdir(maskfolderpath)
        for maskfile in mask_list:
            maskfilepath = maskfolderpath + maskfile
            mask = cv2.imread(maskfilepath, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, size)
            mask_total = mask_total + mask
    column = np.round(np.sum(mask_total,axis=0))
    row = np.round(np.sum(mask_total,axis=1))
    
    c_0 = np.where(column!=0)[0][0]
    c_1 = np.where(column!=0)[0][-1]
    r_0 = np.where(row!=0)[0][0]
    r_1 = np.where(row!=0)[0][-1]
    prior = np.zeros(size)
    prior[r_0:r_1,c_0:c_1] = 1
    prior = torch.from_numpy(prior).unsqueeze(axis=0)    
    prior = prior.repeat(1,1,1,1)
    #cv2.imwrite(foldername + 'prior.png', np.uint8(prior*255))
    return prior

def save_image(MASK_list, path_list, DICE_list, HD95_list, BETT_list, foldername, modelname):
    N = MASK_list.shape[0]        
    stat_file = open(foldername + '/' + modelname + '_test.csv', 'w+')    
    stat_file.write('DICE, HD95, BETT, path\n')
    stat_file.flush()
    for i in range(N):
        cv2.imwrite(path_list[i], np.uint8(np.round(MASK_list[i,:,:,:]*255)))
        stat_file.write('{}, {}, {}, {}\n'.format(DICE_list[i], HD95_list[i], BETT_list[i], path_list[i]))
        stat_file.flush()            
    stat_file.close()

class KiTS2dataset(Dataset):
    def __init__(self, root, phase, model, transform):
        self.phase=phase
        self.path = root + phase
        self.img_list = os.listdir(self.path + 'im2d')
        self.model = model
        self.transform = transform
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index):
        imagename = self.img_list[index]
        path1 = self.path + 'im2d/' + imagename
        path2 = self.path + 'mask/' + imagename
        paths = self.path + self.model + '/' + imagename
        sub_folder = os.path.split(paths)[0]
        if not os.path.exists(sub_folder) and self.phase=='test/':
            os.makedirs(sub_folder)
            print("The new directory is created!   " + sub_folder)
        image = read_image(path1).to(dtype=torch.float32)/255
        mask = read_image(path2).to(dtype=torch.float32)/255
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask, paths



def ims2nps(foldername, full_size, channel):
    IM_list = []
    file_list = os.listdir(foldername)
    for file in file_list:
        print(file)
        if os.path.splitext(file)[1] == '.png' or os.path.splitext(file)[1] == '.jpg':
            im_np = cv2.imread(foldername+file)
            im_np = cv2.resize(im_np,full_size)
            if im_np.ndim == 2:
                im_np = np.expand_dims(im_np, axis=(0,3))
            else:
                im_np = np.expand_dims(im_np, axis=0)
            IM_list.append(im_np)
    IM_array = np.concatenate(IM_list,axis=0)
    IM_array = np.transpose(IM_array, (0,3,1,2))
    return IM_array

def HAM2NPY(grid_size,folder):    
    foldername1 = folder + '/HAM10000_images_part_1/'
    foldername2 = folder + '/HAM10000_images_part_2/'
    foldername_m = folder + '/HAM10000_segmentations_lesion_tschandl/'
    im_array1 = ims2nps(foldername1,grid_size, 3)
    im_array2 = ims2nps(foldername2,grid_size, 3)
    mask_array = ims2nps(foldername_m,grid_size, 1)
    im_array = np.concatenate((im_array1,im_array2),axis=0)
    np.save(folder+'HAM_image',im_array)
    np.save(folder+'HAM_mask',mask_array)
    
def load_skinMNIST(foldername, perc = 1.00):     
    im = np.load(foldername + 'HAM_image_CCA.npy')
    mask = np.load(foldername + 'HAM_mask_CCA.npy')
    im = (im/255)
    print(im.shape)
    print(mask.shape)

    im_train = torch.from_numpy(im[1:int(perc*im.shape[0]),:,:,:])
    mask_train = torch.from_numpy(mask[0:int(perc*im.shape[0]),:,:,:])
    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:])
    trainset = TensorDataset(im_train, im_train[:,0:1,:,:], mask_train)
    testset = TensorDataset(im_test, im_test[:,0:1,:,:], mask_test)           
    
    I_prior = DiskDrawer2D([im.shape[2],im.shape[3]])
    I_prior = torch.from_numpy(I_prior).unsqueeze(axis=0)
    thetaTPS = torch.tensor([[[1,0,0],[0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,2,im.shape[2],im.shape[3]))[0]        
    coor = coor.permute((2,0,1))
    print("------------ LOAD COMPLETE ------------")
    return trainset, testset, I_prior, coor

def load_skinMNISTone(foldername, perc = 1.00, i=0):     
    im = np.load(foldername + 'HAM_image_CCA.npy')
    mask = np.load(foldername + 'HAM_mask_CCA.npy')
    im = (im/255)
    print(im.shape)
    print(mask.shape)

    im_train = torch.from_numpy(im[i:i+1,:,:,:])
    mask_train = torch.from_numpy(mask[i:i+1,:,:,:])
    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:])
    trainset = TensorDataset(im_train, im_train[:,0:1,:,:], mask_train)
    testset = TensorDataset(im_test, im_test[:,0:1,:,:], mask_test)           
    
    I_prior = DiskDrawer2D([im.shape[2],im.shape[3]])
    I_prior = torch.from_numpy(I_prior).unsqueeze(axis=0)
    thetaTPS = torch.tensor([[[1,0,0],[0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,2,im.shape[2],im.shape[3]))[0]        
    coor = coor.permute((2,0,1))
    print("------------ LOAD COMPLETE ------------")
    return trainset, testset, I_prior, coor


def load_ACDC2D(foldername, perc = 0.90):  
    im = np.load(foldername + 'ACDC2_image.npy')
    mask = np.load(foldername + 'ACDC2_mask.npy')
    mask_fill = np.load(foldername + 'ACDC2_mask_fill.npy')
    im = (im/255)
    #mask_fill = mask
    #for i in range(mask.shape[0]):
    #    mask_this = mask[i,0,:,:]
    #    mask_fill[i,0,:,:] = HoleFiller2D(mask_this)        
    print(im.shape)
    print(mask.shape)
    im_train = torch.from_numpy(im[0:int(perc*im.shape[0]),:,:,:])
    mask_train = torch.from_numpy(mask[0:int(perc*im.shape[0]),:,:,:])
    maFi_train = torch.from_numpy(mask_fill[0:int(perc*im.shape[0]),:,:,:])
    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:])
    maFi_test = torch.from_numpy(mask_fill[int(perc*im.shape[0]):,:,:,:])
    trainset = TensorDataset(im_train, mask_train, maFi_train)
    testset = TensorDataset(im_test, mask_test, maFi_test)     

    I_disk = DiskDrawer2D([im.shape[2],im.shape[3]])
    I_circular = HoleDigger2D(I_disk)
    I_disk = torch.from_numpy(I_disk).unsqueeze(axis=0)
    I_circular = torch.from_numpy(I_circular).unsqueeze(axis=0)
    thetaTPS = torch.tensor([[[1,0,0],[0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,2,im.shape[2],im.shape[3]))[0]        
    coor = coor.permute((2,0,1))
    print("------------ LOAD COMPLETE ------------")
    return trainset, testset, I_disk, I_circular, coor

def HoleFiller2D(BW_hole):
    BW_fill = BW_hole
    h,w = BW_hole.shape    
    for i in range(h):
        for j in range(w):
            u = np.sum(BW_hole[i,0:j])
            d = np.sum(BW_hole[i,j+1:])
            l = np.sum(BW_hole[0:i,j])
            r = np.sum(BW_hole[i+1:,j])
            if BW_hole[i,j] == 0 and u>0 and d>0 and l>0 and r>0:
                BW_fill[i,j] = 1
    return BW_fill
def HoleDigger2D(I_disk):
    I_circular = I_disk
    h,w = I_disk.shape
    xc,yc = np.round((h-1)/2),np.round((w-1)/2)
    for i in range(h):
        for j in range(w):
            if np.linalg.norm([(i-xc),(j-yc)]) < 0.1*np.min([h,w]):
                I_circular[i,j] = 0
    return I_circular
def DiskDrawer2D(size):
    h,w = size
    xc,yc = np.round((h-1)/2),np.round((w-1)/2)
    I_disk = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            if np.linalg.norm([(i-xc),(j-yc)]) < 0.25*np.min([h,w]):
                I_disk[i,j] = 1
    return I_disk


def augment_deform(im, mask, rotation, padding_width, scale_ranges = [0.8,1.25]):
    '''
    HINT : 
    1.The parameter 'tran_rate' and 'l' is the same for 'h' and 'w', but can be specific if one want to improve this;
    2.The parameter 'scale_ranges' can be determined automatically by 'h' 'w' and 'padding_width';
    '''    
    batch_size,c,h,w = im.shape
    tran_rate = (padding_width) / max([h,w])
    a=(np.random.rand(batch_size,1,1)-.5)*2*rotation
    l = scale_ranges[1] - scale_ranges[0]
    s1 = (np.random.rand(batch_size,1,1)*l)+scale_ranges[0]
    s2 = (np.random.rand(batch_size,1,1)*l)+scale_ranges[0]
    t1=(np.random.rand(batch_size,1,1)-.5)*2* (tran_rate*2 + (1-tran_rate*2)*(1-s1)) # OR [h/2*(1-s1) + (tran_rate*h*s1)]
    t2=(np.random.rand(batch_size,1,1)-.5)*2* (tran_rate*2 + (1-tran_rate*2)*(1-s2))
    s1 = 1./s1
    s2 = 1./s2
    affine_matrix = torch.from_numpy(np.concatenate(
        (np.concatenate((s1*np.cos(a), -s2*np.sin(a), t1), axis = 2),
            np.concatenate((s1*np.sin(a), s2*np.cos(a), t2), axis = 2)),
            axis = 1))
    affine_matrix = affine_matrix.float()
    mapping = F.affine_grid(affine_matrix, (batch_size,c,h,w), align_corners=True)
    mapping_m = F.affine_grid(affine_matrix, (batch_size,1,h,w), align_corners=True)
    im_deformed = F.grid_sample(im, mapping, mode='bilinear', padding_mode='border', align_corners=True)
    mask_deformed = F.grid_sample(mask, mapping_m, mode='bilinear', padding_mode='border', align_corners=True)
    return im_deformed,mask_deformed    


def visualize_re(IM, MASK, MASK_PRED, MAPPING, file_name):
    IM = IM.data.cpu().numpy()
    IM = (np.transpose( IM, (0,2,3,1))*0.5 + 0.5)*255
    MASK = MASK.data.cpu().numpy()
    MASK = (np.transpose(MASK, (0,2,3,1)))*255
    MASK_PRED = MASK_PRED.data.cpu().numpy()
    MASK_PRED = (np.transpose( MASK_PRED, (0,2,3,1)))*255
    MAPPING = MAPPING.data.cpu().numpy()
    n = IM.shape[0]
    for i in range(n):        
        cv2.imwrite('result_IM/' + file_name + str(i) + '_IM.jpg', IM[i,:,:,:])
        cv2.imwrite('result_IM/' + file_name + str(i) + '_MASK.jpg', MASK[i,:,:,:])
        cv2.imwrite('result_IM/' + file_name + str(i) + '_PRED.jpg', MASK_PRED[i,:,:,:])
        np.save('result_IM/' + file_name + str(i) + '.npy',MAPPING[i,:,:,:])
        
def vec2cor2D(vector, original):
    vector[:,:,:,[0,-1]] = 0
    vector[:,:,[0,-1],:] = 0
    sour_coor = vector + original
    sour_coor = sour_coor.permute((0, 2, 3, 1))
    return sour_coor

def CCAselect(foldername):
    IM = np.load(foldername + 'HAM_image.npy')
    MASK = np.load(foldername + 'HAM_mask.npy')
    print(IM.shape)
    print(MASK.shape)
    N=MASK.shape[0]
    MASK = np.transpose(MASK,(0,2,3,1))
    MASK_list = []
    IM_list = []
    for i in range(N):
        mask = MASK[i,:,:,:]
        mask=cv2.cvtColor(MASK[i,:,:,:], cv2.COLOR_BGR2GRAY)
        mask = np.rint(mask/255).astype(int)
        _,betty_no = label(mask,return_num=True)
        if betty_no == 1:
            MASK_list.append(np.expand_dims(mask, axis = (0,1)))
            IM_list.append(IM[i:i+1,:,:,:])
    im_array = np.concatenate(IM_list,axis=0)
    mask_array = np.concatenate(MASK_list,axis=0)
    print(im_array.shape)
    print(mask_array.shape)
    np.save(foldername+'HAM_image_CCA',im_array)
    np.save(foldername+'HAM_mask_CCA',mask_array.astype(np.uint8))

def save_pair(IM,MASK,path,count):    
    IM = np.transpose(IM, (0,2,3,1))
    MASK = np.transpose(MASK, (0,2,3,1))
    N=IM.shape[0]
    for i in range(N):
        cv2.imwrite(path + '%d_%dIM.jpg'%(count,i), 255*IM[i,:,:,:])
        cv2.imwrite(path + '%d_%dMASK.jpg'%(count,i), 255*np.round(MASK[i,:,:,:]))

def plot_grid(grid, path,count):
    GRID = grid
    N=grid.shape[0]
    for i in range(N):
        fig = plt.figure(figsize=(16, 16), dpi=150)
        segs1f = GRID[i,:,:,:]
        segs2f = segs1f.transpose(1, 0, 2)
        plt.gca().add_collection(LineCollection(segs1f))
        plt.gca().add_collection(LineCollection(segs2f))
        plt.gca().axis('equal')
        fig.savefig(path + '%d_%dGRID.png'%(count,i))
        plt.close()

if __name__ == '__main__':
    HAM2NPY([128,128],folder = 'C:/Users/Han/Desktop/TPSN/DATA_Ham/')
    CCAselect(foldername = 'C:/Users/Han/Desktop/TPSN/DATA_Ham/')