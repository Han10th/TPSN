import os
import sys
import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torchvision.transforms.functional as FF

from utils.buildingblocks_teds import ConvBlock,EncoderBranch,DecoderBranch,BottleNeck
from utils.utils_teds import WholeDiffeoUnit

class TEDS_Net(nn.Module):
    '''
    TEDS-Net:
    Input is the parameter describing dictionary.
    '''

    def __init__(self, in_chan, out_chan, ndims, inshape):
        
        
        super(TEDS_Net, self).__init__()


        # Parameters settings - arch:
        in_channels = in_chan
        out_channels = out_chan
        features = 6
        net_depth= 4
        dropout = 0.2
        dec_depth = [1,1]
        self.no_branches = 2
        
        # Parameters settings - diffeomorphic:
        int_steps = 7
        GSmooth =1
        Guas_kernel = 5
        Guas_P = 2
        act = True
        self.mega_P = 2

        # Dataset dependant parameters
        ndims = ndims
        inshape = inshape
        

        # -------------------------------------------------------------------
        # --------- 1. Enc:
        self.enc = EncoderBranch(in_channels,features,ndims,net_depth,dropout)
        
        # --------- 2. Bottleneck:
        self.bottleneck = BottleNeck(features,ndims,net_depth,dropout)

        # --------- 3. Decoder + Diffeo Units:
        if self.no_branches ==1:
            self.STN = WholeDiffeoUnit(out_channels,ndims,inshape,branch=0)
        elif self.no_branches ==2:
            self.STN_bulk = WholeDiffeoUnit(out_channels,ndims,inshape,branch=0)
            self.STN_ft = WholeDiffeoUnit(out_channels,ndims,inshape,branch=1)

        # --------------------------------------------------------------------
        # --------- 4. Downsample to up-sampled fields (visualisation):
        if self.mega_P>1:
            if ndims ==2:
                from torch.nn import MaxPool2d as MaxPool
            elif ndims ==3:
                from torch.nn import MaxPool3d as MaxPool
            self.downsample =MaxPool(kernel_size= 3,stride =self.mega_P,padding=1) # downsample the final results


    def forward(self, x,prior_shape):
        '''
        Inputs [ tensor] - dims [Batch,2,Chan,X,Y,Z] where the second dimension has both the prior shape and the image
        '''

        # -------- 0. Get inputs
        #x = inputs[:,0,...] # first channel of batch
        #prior_shape =inputs[:,1:] # second channel of batch
 
        # -------- 1. Enc + Bottleneck:
        enc_outputs = self.enc(x)
        BottleNeck =self.bottleneck(enc_outputs[-1])

        # --------- 2. Dec + Diffeo:
        if self.no_branches ==1:
            flow_field,flow_upsamp,sampled=self.STN(BottleNeck,enc_outputs,prior_shape)
            # DOWNSAMPLE
            if self.mega_P>1:
                sampled = self.downsample(sampled)

            return sampled,flow_upsamp,flow_field

        elif self.no_branches ==2:
            flow_bulk_field,flow_bulk_upsamp,bulk_sampled=self.STN_bulk(BottleNeck,enc_outputs,prior_shape)
            flow_ft_field,flow_ft_upsamp,ft_sampled=self.STN_ft(BottleNeck,enc_outputs,bulk_sampled)

            if self.mega_P>1:
                bulk_sampled = self.downsample(bulk_sampled)
                ft_sampled = self.downsample(ft_sampled)
            

            return ft_sampled, flow_bulk_upsamp,flow_ft_upsamp
        
