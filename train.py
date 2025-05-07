from ultralytics import YOLO


def train():
    model = YOLO(model="/home/jovyan/work/detection_nn/yolov8m.pt")
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print(model.info())
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    results = model.train(
        data="data.yaml",
        epochs=500,
        imgsz=640,
        batch=26,
        save=True,
        save_period=1,
        device="0",
        optimizer="AdamW",
        patience=100,
        val=True,
        augment=True,
        cos_lr=True,
        project="runs/detect",
        name="yolov8m_custom",
        pretrained="/home/jovyan/work/detection_nn/yolov8m.pt",
    )

    print(model.info())


train()
