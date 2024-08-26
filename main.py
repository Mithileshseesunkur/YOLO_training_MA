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

    model = YOLO(r"C:\Users\mseesunker\Desktop\YOLO\trained_model\MA_model-20240819T084430Z-001\MA_model\weights\last.pt")

    pathCleanedDataset = r"C:\Users\mseesunker\Desktop\YOLO\trained_model\MA_model-20240819T084430Z-001\train_yolo_continue\YOLO_training_MA\data.yaml"
    pathModel = r"C:\Users\mseesunker\Desktop\YOLO\trained_model\MA_model-20240819T084430Z-001\MA_model_continue"

    results = model.train(
        data=pathCleanedDataset,
        resume=True,
        name="MA_model",
        project=pathModel,
        save=True,
        device=0,
        epoch=200,
        #patience=50,
        batch=16,
        save_dir=r"C:\Users\mseesunker\Desktop\YOLO\trained_model\MA_model-20240819T084430Z-001\MA_model"
    )

if __name__ == '__main__':
    main()

