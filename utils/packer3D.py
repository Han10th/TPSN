import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat,savemat
from torch.utils.data import TensorDataset


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
        np.save('result_IM/' + file_name + str(i) + '.npy',MAPPING[i,:,:,:])
        
def load_bctv(foldername, perc = 0.85, noclass=8):
    im = np.load(foldername + 'bctv_im.npy')#[0:2,:,:,:,:]
    mask = np.load(foldername + 'bctv_mask.npy')#[0:2,:,:,:,:]
    prior = np.load(foldername + 'bctv_prior.npy')#[0:2,:,:,:,:]
    im = np.concatenate((im,prior),axis=1)
    print(im.shape)
    print(mask.shape)
    print(prior.shape)
    im_train = torch.from_numpy(im[0:int(perc*im.shape[0]),:,:,:,:])
    mask_train = torch.from_numpy(mask[0:int(perc*im.shape[0]),:,:,:,:])
    prior_train = torch.from_numpy(prior[0:int(perc*im.shape[0]),:,:,:,:])
    im_train, mask_train,prior_train = im_train.float(), mask_train.float(),prior_train.float()
    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:,:])
    prior_test = torch.from_numpy(prior[int(perc*im.shape[0]):,:,:,:,:])
    im_test, mask_test, prior_test = im_test.float(), mask_test.float(), prior_test.float()
    trainset = TensorDataset(im_train, mask_train,prior_train)
    testset = TensorDataset(im_test, mask_test,prior_test)    
    
    thetaTPS = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,1,im.shape[2],im.shape[3],im.shape[4]), align_corners=True)[0]        
    coor = coor.permute((3,0,1,2))  
    print("------------ LOAD COMPLETE ------------")
    return trainset, testset, coor.repeat(noclass,1,1,1)
def load_medbrain(foldername, perc = 0.87, noclass=8):
    im = np.load(foldername + 'midbrain_im.npy')#[0:2,:,:,:,:]
    mask = np.load(foldername + 'midbrain_mask.npy')#[0:2,:,:,:,:]
    print(im.shape)
    print(mask.shape)
    im_train = torch.from_numpy(im[0:int(perc*im.shape[0]),:,:,:,:])
    mask_train = torch.from_numpy(mask[0:int(perc*im.shape[0]),:,:,:,:])
    im_train, mask_train = im_train.float(), mask_train.float()
    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:,:])
    im_test, mask_test = im_test.float(), mask_test.float()
    trainset = TensorDataset(im_train, mask_train)
    testset = TensorDataset(im_test, mask_test)    
    
    thetaTPS = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,1,im.shape[2],im.shape[3],im.shape[4]), align_corners=True)[0]        
    coor = coor.permute((3,0,1,2))
    mask_template = np.load(foldername + 'midbrain_maskprior.npy')       
    mask_template = torch.from_numpy(mask_template)
    print("------------ LOAD COMPLETE ------------")
    return trainset, testset, mask_template, coor.repeat(noclass,1,1,1)
def DiskDrawer3D(size):
    d,h,w = size
    xc,yc,zc = np.round((d-1)/2),np.round((h-1)/2),np.round((w-1)/2)
    I_disk = np.zeros([1,d,h,w])
    for k in range(d):
        for i in range(h):
            for j in range(w):
                if np.linalg.norm([(i-xc),(j-yc),(j-zc)]) < 0.25*np.min([d,h,w]):
                    I_disk[0,k,i,j] = 1.0
    return I_disk

def load_kits21(foldername, perc = 0.70):
    im = np.load(foldername + 'im_kits21.npy')#
    mask = np.load(foldername + 'mask_kits21.npy')    
    print(im.shape)
    print(mask.shape)

    im_train = torch.from_numpy(im[0:int(perc*im.shape[0]),:,:,:,:])
    mask_train = torch.from_numpy(mask[0:int(perc*im.shape[0]),:,:,:,:])
    im_train, mask_train = im_train.float(), mask_train.float()
    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:,:])
    im_test, mask_test = im_test.float(), mask_test.float()
    trainset = TensorDataset(im_train, mask_train)
    testset = TensorDataset(im_test, mask_test)    
    
    thetaTPS = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,1,im.shape[2],im.shape[3],im.shape[4]), align_corners=True)[0]        
    coor = coor.permute((3,0,1,2))
    mask_template = mask_train[180,:,:,:,:]      
    print("------------ LOAD COMPLETE ------------")
    return trainset, testset, mask_template, coor
def load_kits21_tumor(foldername, perc = 0.70):
    im = np.load(foldername + 'im_kits21.npy')#
    mask = np.load(foldername + 'mask_kits21_tumor.npy')    
    print(im.shape)
    print(mask.shape)

    im_train = torch.from_numpy(im[0:int(perc*im.shape[0]),:,:,:,:])
    mask_train = torch.from_numpy(mask[0:int(perc*im.shape[0]),:,:,:,:])
    im_train, mask_train = im_train.float(), mask_train.float()
    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:,:])
    im_test, mask_test = im_test.float(), mask_test.float()
    trainset = TensorDataset(im_train, mask_train)
    testset = TensorDataset(im_test, mask_test)    
    
    thetaTPS = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,1,im.shape[2],im.shape[3],im.shape[4]), align_corners=True)[0]        
    coor = coor.permute((3,0,1,2))
    mask_template = torch.from_numpy(np.load(foldername + 'template3d.npy')[0])
    mask_template = mask_template.float()
    print("------------ LOAD COMPLETE ------------")
    return trainset, testset, mask_template, coor

def load_kits21_test(foldername, perc = 0.70):       
    im = np.load(foldername + 'im_kits21.npy')#
    mask = np.load(foldername + 'mask_kits21.npy')    
    print(im.shape)
    print(mask.shape)

    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:,:])
    im_test, mask_test = im_test.float(), mask_test.float()
    testset = TensorDataset(im_test, mask_test)    
    
    thetaTPS = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,1,im.shape[2],im.shape[3],im.shape[4]), align_corners=True)[0]        
    coor = coor.permute((3,0,1,2))
    mask_template = torch.from_numpy(mask[180,:,:,:,:])     
    print("------------ LOAD COMPLETE ------------")
    return testset, mask_template, coor

def load_kits21_interruption(foldername, perc = 0.70):
    im = np.load(foldername + 'im_kits21.npy')#
    mask = np.load(foldername + 'mask_kits21.npy')
    band_depth = 10
    width = int(32 - band_depth/2)
    im[:,:,width:64-width,:,:] = 0
    print(im.shape)
    print(mask.shape)

    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:,:])
    im_test, mask_test = im_test.float(), mask_test.float()
    testset = TensorDataset(im_test, mask_test)    
    
    thetaTPS = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,1,im.shape[2],im.shape[3],im.shape[4]), align_corners=True)[0]        
    coor = coor.permute((3,0,1,2))
    mask_template = torch.from_numpy(mask[180,:,:,:,:])        
    print("------------ LOAD COMPLETE ------------")
    return testset, mask_template, coor

def load_kits21_predU(foldername):
    im_test, mask_test = im_test.float(), mask_test.float()
    testset = TensorDataset(im_test, mask_test)    
    
    thetaTPS = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,1,im.shape[2],im.shape[3],im.shape[4]), align_corners=True)[0]        
    coor = coor.permute((3,0,1,2))
    mask_template = torch.from_numpy(mask[180,:,:,:,:])        
    print("------------ LOAD COMPLETE ------------")
    return testset, mask_template, coor


def load_kits21_HYBT(foldername, stage=0, perc = 0.70):
    im = np.load(foldername + 'im_kits21.npy')#
    #immaU = np.load(foldername + 'maskU_kits21.npy')#
    immaT = np.load(foldername + 'maskT' + str(stage) + '_kits21.npy')#
    mask = np.load(foldername + 'mask_kits21.npy')    
    im = np.concatenate((im,immaT.astype(float)),axis = 1)
    print(im.shape)
    print(mask.shape)

    im_train = torch.from_numpy(im[0:int(perc*im.shape[0]),:,:,:,:])
    mask_train = torch.from_numpy(mask[0:int(perc*im.shape[0]),:,:,:,:])
    im_train, mask_train = im_train.float(), mask_train.float()
    im_test = torch.from_numpy(im[int(perc*im.shape[0]):,:,:,:,:])
    mask_test = torch.from_numpy(mask[int(perc*im.shape[0]):,:,:,:,:])
    im_test, mask_test = im_test.float(), mask_test.float()
    trainset = TensorDataset(im_train, mask_train)
    testset = TensorDataset(im_test, mask_test)    
    
    thetaTPS = torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]], dtype=torch.float)        
    coor = F.affine_grid(thetaTPS, (1,1,im.shape[2],im.shape[3],im.shape[4]), align_corners=True)[0]        
    coor = coor.permute((3,0,1,2))
    mask_template = mask_train[180,:,:,:,:]      
    print("------------ LOAD COMPLETE ------------")
    return trainset, testset, mask_template, coor

def vec2cor3D(vector, original):
    vector[:,:,:,:,[0,-1]] = 0
    vector[:,:,:,[0,-1],:] = 0
    vector[:,:,[0,-1],:,:] = 0
    sour_coor = vector + original
    sour_coor = sour_coor.permute((0, 2, 3, 4, 1))
    return sour_coor
def save_mat(any_list, name):
    any_array = np.concatenate(any_list, axis = 0)
    #np.save('maskT_kits21.npy', np.round(any_array).astype(np.int16))
    mdic = {name: np.round(any_array).astype(np.int16)}
    savemat( name + '.mat', mdic)
def save_np(any_list, name):
    any_array = np.concatenate(any_list, axis = 0)
    np.save(name, np.round(any_array).astype(np.int16))
def save_npasmat(any_np, name):
    mdic = {name: any_np}
    savemat( name + '.mat', mdic)

def normalise(image):
    # normalise and clip images -1000 to 800
    np_img = image
    np_img = np.clip(np_img, -1000., 800.).astype(np.float32)
    return np_img
def whitening(image):
    """Whitening. Normalises image to zero mean and unit variance."""
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret
def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""
    image = image.astype(np.float32)
    minimum = np.min(image)
    maximum = np.max(image)
    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret
def normalise_one_one(image):
    """Image normalisation. Normalises image to fit [-1, 1] range."""
    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret


#if __name__ == '__main__':
#    im_list = []
#    mask_list = []
#    path = 'C:\\Users\\hzhang\\Desktop\\KiTS21\\mat_data\\'
#    for i in range(300):
#        im_path = 'case_%05d_im' % i
#        mask_path = 'case_%05d_mask' % i
#        im_file = loadmat(path+im_path)
#        mask_file = loadmat(path+mask_path)
#        im = normalise_zero_one(im_file['im3d_resize'])
#        mask = mask_file['mask_resize']
#        im_list.append(np.expand_dims(im, axis=(0,1)))
#        mask_list.append(np.expand_dims(mask, axis=(0,1)))
#    im_array = np.concatenate(im_list, axis=0)
#    mask_list = np.concatenate(mask_list, axis=0)
#    print(im_array.shape)
#    print(mask_list.shape)
#    np.save('im_kits21.npy', im_array)
#    np.save('mask_kits21.npy', mask_list)


if __name__ == '__main__':
    foldername = 'C:\\Users\\Han\\Desktop\\TPSN3D\\DATA\\'
    im_kits21 = np.load(foldername + 'im_kits21.npy')#
    mask_kits21 = np.load(foldername + 'mask_kits21.npy')    
    path = 'D:\\PROJECT\\kits21\\kits21\\data\\mat_data\\'
    for i in [14,18,42,166,40,151,293]:
        im_path = 'case_%05d_im' % i
        mask_path = 'case_%05d_mask' % i
        im_file = loadmat(path+im_path)
        mask_file = loadmat(path+mask_path)
        im = normalise_zero_one(im_file['im3d_resize'])
        mask = mask_file['mask_resize']
        im_kits21[i,0,:,:,:] = im
        mask_kits21[i,0,:,:,:] = mask
    print(im_kits21.shape)
    print(mask_kits21.shape)
    np.save('im_kits21_fixed.npy', im_kits21)
    np.save('mask_kits21_fixed.npy', mask_kits21)
    save_npasmat(im_kits21,'im_kits21')
    save_npasmat(mask_kits21,'mask_kits21')

#from scipy.io import loadmat,savemat
#import numpy as np 
#im_array = loadmat('D:\\PROJECT\\DATA\\BCTV\\bctv_im.mat')['im_array']
#mask_array = loadmat('D:\\PROJECT\\DATA\\BCTV\\bctv_mask.mat')['mask_array']
#prior_array = loadmat('D:\\PROJECT\\DATA\\BCTV\\bctv_prior.mat')['prior_array']
#np.save('D:\\PROJECT\\DATA\\BCTV\\bctv_im.npy',im_array)
#np.save('D:\\PROJECT\\DATA\\BCTV\\bctv_mask.npy',mask_array)
#np.save('D:\\PROJECT\\DATA\\BCTV\\bctv_prior.npy',prior_array)