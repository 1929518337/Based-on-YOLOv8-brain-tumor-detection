from ultralytics import YOLO

# 加载模型
model = YOLO("D:\\yolov8-8.2.36\\yolov8m.pt")
model = YOLO("D:\\yolov8-8.2.36\\yolov8m.pt")

# Use the model
# results = model.train(data="ultralytics/datasets/rain.yaml", epochs=20, batch=-1)  # 训练模型
if __name__ == '__main__':
    # Use the model
    model.train(model="D:\\yolov8-8.2.36\\yolov8m.pt", data="D:\\yolov8-8.2.36\\dataset-Br35H\\br35h.yaml", epochs=300, imgsz=640, batch=1)

    results = model.val()
    # results = model("自己的验证图片")
    success = YOLO("yolov8n.pt").export(format="onnx")