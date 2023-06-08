# @author: hayat
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dehazeNet_module.image_data_loader as image_data_loader_from_image_data_loaderpy
import dehazeNet_module.lightdehazeNet as lightdehazeNet_object_from_lightdehazeNet
import numpy
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
from matplotlib import pyplot as plt
import cv2

def image_haze_removel(input_image):

	
	hazy_image = (np.asarray(input_image)/255.0)

	hazy_image = torch.from_numpy(hazy_image).float()
	hazy_image = hazy_image.permute(2,0,1)
	hazy_image = hazy_image.cuda().unsqueeze(0)

	ld_net = lightdehazeNet_object_from_lightdehazeNet.LightDehaze_Net().cuda()
	ld_net.load_state_dict(torch.load('dehazeNet_Module/trained_weights/trained_LDNet.pth'))

	dehaze_image = ld_net(hazy_image)
	return dehaze_image
	
		
