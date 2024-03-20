import numpy as np
def dice(mask1, mask2):
    epsilon = 1e-5    
    #assert mask1.size() == mask2.size(), "the size of predict and target must be equal."
    #num = predict.size(0)        
    dims = (0, 1, 2, 3)
    intersection = np.sum(mask1 * mask2, dims)
    cardinality = np.sum(mask1 + mask2, dims)
    dice_score = 2. * intersection / (cardinality + epsilon)
    return np.mean(1. - dice_score)


foldername = 'C:\\Users\\Han\\Desktop\\TPSN3D\\'
#im = np.load(foldername + 'im_kits21.npy')
mask = np.load(foldername + 'mask_kits21.npy')
mask = mask.astype(np.float32)

score = np.zeros([300,300])
for i in range(300):
    this_mask = mask[i,:,:,:,:]
    for j in range(300):
        if i!=j:
            that_mask = mask[j,:,:,:,:]
            score[i,j] = dice(this_mask,that_mask)
np.savetxt("dice_scnpore.csv", score, delimiter=",")
max_dice = score.max()
index = np.where(score == max_dice)
print("The max dice is %.4f" % (max_dice))
print("The index is :")
print(index)

