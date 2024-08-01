import os
import splitfolders as spl

#defining paths

source_dir=r"C:\Users\mitarbeiter\PycharmProjects\cleaned_dataset"
output_dir=r"C:\Users\mitarbeiter\PycharmProjects\split_dataset"


#splitting images at random

spl.ratio(source_dir,output=output_dir,seed=42, ratio=(0.7,0.2,0.1),group_prefix=None)