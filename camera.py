import argparse
import os

import numpy
import numpy as np
import cv2
import sys
sys.path.append("yolo_mask")
import torch
import torchvision
import darknet_video
from darknet_video import yolo_detector
from models.pfld import PFLDInference, AuxiliaryNet
from darknet import bbox2points
from mtcnn.detector import detect_faces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def Euclidean_Distance(vec1,vec2):
    dist = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))
    return dist

def main(args):

    checkpoint = torch.load(args.model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])

    pfld_backbone.eval()
    pfld_backbone = pfld_backbone.to(device)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        # img=cv2.imread('1.jpg')
        if not ret: break
        # height, width = img.shape[:2]
        # bounding_boxes, landmarks = detect_faces(img)
        detections,img,width,height=yolo_detector(img)
        for label, confidence, bbox in detections:
            x1, y1, x2, y2 = bbox2points(bbox)
            # cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
            # cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
            #             (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             colors[label], 2)
            # x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
            # x10, y10, x20, y20 = (box[:4] + 0.5).astype(np.int32)
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            cx = x1 + w // 2
            cy = y1 + h // 2

            size = int(max([w, h]) * 1.1)
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            edx1 = max(0, -x1)
            edy1 = max(0, -y1)
            edx2 = max(0, x2 - width)
            edy2 = max(0, y2 - height)

            cropped = img[y1:y2, x1:x2]
            if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
                cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                             cv2.BORDER_CONSTANT, 0)

            input = cv2.resize(cropped, (112, 112))
            input = transform(input).unsqueeze(0).to(device)
            _, landmarks = pfld_backbone(input)
            pre_landmark = landmarks[0]

            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(
                -1, 2) * [size, size] - [edx1, edy1]
            pre_landmark = numpy.array(pre_landmark)
            normal_landmark = (pre_landmark - pre_landmark.min()) / (pre_landmark.max() - pre_landmark.min())
            min_distance=9999
            txtname=""
            for txt in os.listdir('./test_txt'):
                landmark=numpy.loadtxt("./test_txt/"+txt)
                distance=Euclidean_Distance(normal_landmark,landmark)
                if distance<min_distance:
                    txtname = txt.split('.')[0]
                    min_distance=distance


            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(img, (x1 + x, y1 + y), 1, (0, 255, 0))
                cv2.rectangle(img,(x1,y1),(x2,y2), (0,255,255),1)
                cv2.putText(img, "{} [{:.2f}]".format(label, float(confidence)),
                            (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,255,255), 2)


                cv2.putText(img, "distance:"+"%.5f"%min_distance,
                            (x1, y2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,255,255), 2)
                # if min_distance < 1:
                cv2.putText(img, "ID="+txtname,
                                ((x2+x1)//2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 2)

        cv2.imshow('face_landmark_68', img)
        if cv2.waitKey(10) == 27:
            break


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path',
                        default="./checkpoint/snapshot/checkpoint_epoch_255.pth-1.tar",
                        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
