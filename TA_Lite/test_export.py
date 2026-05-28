import torch
from train_vehicle_detector import build_model, export_onnx

model = build_model(num_classes=3, pretrained_backbone=False)
checkpoint = torch.load('output/vehicle_detector_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])

export_onnx(model, 'output/vehicle_detector.onnx', 320)
