from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
from net import Darknet
import os.path as osp
import pandas as pd
import random

from torch2trt import torch2trt

'''
Creating Command Line Arguments
'''
def arg_parse():
    '''
    Parse arguments to the detect module
    '''
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")
    parser.add_argument("--images", dest="images", help = "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest = 'det', help = "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1, type=int)
    parser.add_argument("--confidence", dest = "confidence", type=float,
                        help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", type=float,
                        help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = "Config file",
                        default = "cfg\yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile",
                        default = "weights\yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to     run detection on", 
                        default = "video.avi", type = str)
    return parser.parse_args()

def load_classes(namesfile):
    names = []
    with open(namesfile, "r") as fp:
        # ????????????????
        names = fp.read().split('\n')[:-1]

    return names

def write(x, results):
    c1 = tuple(x[1:3].int().tolist())
    c2 = tuple(x[3:5].int().tolist())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

    
if __name__ == "__main__":
    args = arg_parse()
    images = args.images
    batch_size = args.bs
    confidence = args.confidence
    nms_thesh = args.nms_thresh
    start = 0
    CUDA = torch.cuda.is_available()
    #CUDA = False

    TENSORRT = True

    num_classes = 80    #For COCO

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU available, put the model on GPU
    if CUDA:
        model.cuda()
    
    model.eval()

    if TENSORRT:
        x = torch.rand(size=(1, 3,  opt.img_size, opt.img_size)).cuda()

    videofile = "video.avi"
    
    # cap = cv2.VideoCapture(videofile)

    cap = cv2.VideoCapture(0)  # for webcam

    assert cap.isOpened(), 'cannot capture source'

    frames = 0
 
    while cap.isOpened():
        ret,frame = cap.read()
        if ret:
            start = time.time()
            img = prep_image(frame, inp_dim)
            # notice that: im_dim is photo zoom
            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1,2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            with torch.no_grad():
                output = model(img, CUDA)
            
            output = write_results(output, confidence, num_classes, nms_thesh)
           
            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.4f}".format( 1 / (time.time() - start)))
                cv2.imshow("frame", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:, 1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))

            # based on the dimension of input and original image, zoom bbox
            im_dim = im_dim.repeat(output.size(0), 1)/inp_dim
            output[:,1:5] *= im_dim

            classes = load_classes("data/coco.names")
    
            colors = []
            with open("color\pallete", "rb") as fp:
                colors = pkl.load(fp)

            list(map(lambda x: write(x, frame), output))

            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print(time.time() - start)
            print("FPS of the video is {:5.2f}".format( 1 / (time.time() - start)))
        else:
            break     
