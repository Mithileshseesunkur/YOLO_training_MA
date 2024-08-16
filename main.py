from ultralytics import YOLO

def main():
    model = YOLO("yolov10m.yaml")

    pathCleanedDataset = r"C:\Users\mseesunkur\YOLO\cleaning_dataset\data.yaml"
    pathModel = r"C:\Users\mseesunkur\YOLO\training\YOLO_training_MA"

    results = model.train(
        data=pathCleanedDataset,
        resume=True,
        name="MA_model",
        project=pathModel,
        save=True,
        device=(0,1,2),
        patience=50,
        batch=15,
        save_dir=r"C:\Users\mseesunkur\YOLO\training\YOLO_training_MA"
    )

if __name__ == '__main__':
    main()

# import torch
#
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
#     device = torch.device("cpu")
#
# print(f"Using device: {device}")
#
#
# print(torch.backends.cudnn.version())
