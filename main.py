from ultralytics import YOLO

# import torch
# #import torch.distributed as dist
# #dist.init_process_group(backend='nccl')
# print("libuv support:", torch.distributed.is_available())


# def main():
#     model = YOLO("yolov10m.yaml")

#     pathCleanedDataset = r"C:\Users\mseesunkur\YOLO\cleaning_dataset\data.yaml"
#     pathModel = r"C:\Users\mseesunkur\YOLO\training\YOLO_training_MA"

#     results = model.train(
#         data=pathCleanedDataset,
#         resume=True,
#         name="MA_model",
#         project=pathModel,
#         save=True,
#         device=(0,1,2),
#         patience=50,
#         batch=15,
#         save_dir=r"C:\Users\mseesunkur\YOLO\training\YOLO_training_MA"
#     )

# if __name__ == '__main__':
#     main()

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
import os
import torch
import torch.distributed as dist
from ultralytics import YOLO

# def init_distributed():
#     # Set environment variables
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '50311'
#     os.environ['WORLD_SIZE'] = '3'  # Number of processes
#     os.environ['RANK'] = '0'  # Rank of the current process

#     # Initialize the distributed process group
#     dist.init_process_group(backend='gloo')

def main():
    #init_distributed()

    model = YOLO(r"C:\Users\mseesunkur\YOLO\training\YOLO_training_MA\MA_model3\weights\last.pt")

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

