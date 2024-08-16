import os
import splitfolders as spl

#defining paths

source_dir=r"C:\Users\mseesunkur\YOLO\cleaning_dataset\cleaned"
output_dir=r"C:\Users\mseesunkur\YOLO\cleaning_dataset\cleaned\split"


#splitting images at random

spl.ratio(source_dir,output=output_dir,seed=42, ratio=(0.7,0.2,0.1),group_prefix=None)