from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx", dynamic=False, simplify=True, opset=12, imgsz=(640, 640))
