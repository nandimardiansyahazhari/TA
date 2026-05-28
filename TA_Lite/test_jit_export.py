import torch
from train_vehicle_detector import build_model, _SSDWrapper

model = build_model(num_classes=3, pretrained_backbone=False)
checkpoint = torch.load('output/vehicle_detector_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])

model.eval()
wrapper = _SSDWrapper(model)
wrapper.eval()

dummy = torch.zeros(1, 3, 320, 320)
traced = torch.jit.trace(wrapper, dummy)

torch.onnx.export(traced, dummy, 'output/vehicle_detector_traced.onnx', opset_version=11)
print("ONNX EXPORT SUCCESS")
