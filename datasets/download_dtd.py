## PROCESS SUN397 DATASET

import os
import shutil
from pathlib import Path


def process_dataset(txt_files, downloaded_data_path, output_folder):
    for txt_file in txt_files:
        txt_file_dir = os.path.join(downloaded_data_path, txt_file)
        with open(txt_file_dir, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            input_path = line.strip()
            # import pdb;pdb.set_trace()
            final_folder_name = "_".join(x for x in input_path.split('/')[:-1])
            filename = input_path.split('/')[-1]
            output_class_folder = os.path.join(output_folder, final_folder_name)
            
            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)
            # import pdb;pdb.set_trace()
            full_input_path = os.path.join(downloaded_data_path,'images', input_path)
            output_file_path = os.path.join(output_class_folder, filename)
            # print(final_folder_name, filename, output_class_folder, full_input_path, output_file_path)
            # exit()
            shutil.copy(full_input_path, output_file_path)
            if i % 100 == 0:
                print(f"Processed {i}/{len(lines)} images")

downloaded_data_path = "./data/dtd"
process_dataset(['labels/train1.txt','labels/val1.txt'], downloaded_data_path, os.path.join(downloaded_data_path, "train"))
process_dataset(['labels/test1.txt'], downloaded_data_path, os.path.join(downloaded_data_path, "val"))