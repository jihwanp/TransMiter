import os
import shutil

def create_directory_structure(data_root, split):
    split_file = f'resisc45-{split}.txt'
    with open(os.path.join(data_root, split_file), 'r') as f:
        lines = f.readlines()
    for l in lines:
        l = l.strip()
        class_name = '_'.join(l.split('_')[:-1])
        class_dir = os.path.join(data_root, 'NWPU-RESISC45', class_name)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        src_path = os.path.join(data_root, 'NWPU-RESISC45', l)
        dst_path = os.path.join(class_dir, l)
        print(src_path, dst_path)
        shutil.move(src_path, dst_path)

data_root = './data/resisc45'
for split in ['train', 'val', 'test']:
    create_directory_structure(data_root, split)