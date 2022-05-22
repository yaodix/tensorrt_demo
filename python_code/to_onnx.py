import torchvision.models as models
import torch
import torch.onnx

import onnx

resnet50 = models.resnet50()
resnet50.load_state_dict(torch.load("resnet50-19c8e357.pth"))

BATCH_SIZE=4
dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)
torch.onnx.export(resnet50, dummy_input, "resnet50_pytorch_bs4.onnx", verbose=False)

onnx_model = onnx.load("resnet50_pytorch_bs4.onnx")
onnx.checker.check_model(onnx_model)