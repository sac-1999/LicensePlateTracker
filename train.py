from ultralytics import YOLO
model = YOLO('yolov8n.yaml')
results = model.train(data = './yolov7-license-plate-detection-3/data.yaml',epochs=15)