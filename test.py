import argparse
import os
import time

import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.datasets import WLFWDatasets
import torchvision
from models.pfld import PFLDInference

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate():
    cudnn.benchmark = True
    cudnn.determinstic = True
    cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
    # pfld_backbone.eval()

    pfld_backbone = PFLDInference().to(device)
    checkpoint = torch.load("./checkpoint/snapshot/checkpoint_epoch_255.pth-1.tar", map_location=device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    pfld_backbone.eval()
    for image in os.listdir('pre_test'):
        img=cv2.imread("pre_test/"+image)
        img = cv2.resize(img, (112, 112))


        input = transform(img).unsqueeze(0).to(device)

        pfld_backbone = pfld_backbone.to(device)

        _, landmarks = pfld_backbone(input)
        pre_landmarks = landmarks[0]

        pre_landmarks = pre_landmarks.cpu().detach().numpy()
        pre_landmarks = pre_landmarks.reshape(-1,
                                      2)*[112,112] # landmark



        pre_landmark = numpy.array(pre_landmarks)
        txtstr = "./test_txt/" + image.split('.')[0] + ".txt"
        normal_landmark = (pre_landmark - pre_landmark.min()) / (pre_landmark.max() - pre_landmark.min())
        numpy.savetxt(txtstr, normal_landmark)
        for (x, y) in pre_landmark.astype(np.int32):
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

        cv2.imwrite("./test_image",img)


def main():


    validate()



if __name__ == "__main__":

    main()
