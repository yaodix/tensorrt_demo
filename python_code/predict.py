import cv2
import torch
from albumentations import Resize, Compose
from albumentations.pytorch.transforms import  ToTensorV2
from albumentations.augmentations.transforms import Normalize
from torchvision import models

def preprocess_image(img_path):
    # transformations for the input data
    transforms = Compose([
        Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # read input image
    input_img = cv2.imread(img_path)
    # do transformations
    input_data = transforms(image=input_img)["image"]
    batch_data = torch.unsqueeze(input_data, 0)
    return batch_data



def postprocess(output_data):
    # get class names
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100
    # find top predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    # print the top classes predicted by the model
    while confidences[indices[0][i]] > 0.5:
        class_idx = indices[0][i]
        print(
            "class:",
            classes[class_idx],
            ", confidence:",
            confidences[class_idx].item(),
            "%, index:",
            class_idx.item(),
        )
        i += 1


model = models.resnet50(torch.load("/home/yao/workspace/tensorrt_demo/python_code/resnet50-19c8e357.pth"))
input = preprocess_image("/home/yao/workspace/tensorrt_demo/python_code/Selection_002.jpg").cuda()
model.eval()
model.cuda()
output = model(input)
postprocess(output)