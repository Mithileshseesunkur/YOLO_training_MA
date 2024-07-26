from ultralytics import YOLO

model=YOLO("yolov8m.yaml")


pathCleanedDataset=r"C:\Users\mitarbeiter\PycharmProjects\YOLO_training_MA\model.yaml"
pathModel=r"C:\Users\mitarbeiter\PycharmProjects\model"
results=model.train(data=pathCleanedDataset,resume=True,
                    name="MA_model", project=pathModel,save=True)