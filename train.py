from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter

def train():
    model = YOLO(model="yolov8n.pt")
    print(model.info())
    results = model.train(data="data.yaml", epochs=200, imgsz=640, batch=1, save=True,
                          save_period = 1, device="0", optimizer="NAdam", patience=10, val=True, augment=True)

    log_dir ="runs/detect/yolov8n_custom"


train()