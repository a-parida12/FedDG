import numpy as np
from PIL import Image
import json
import os

def read_json(file):
    print(file)
    with open(file) as f:
        data = json.load(f)
    return data


def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)
    
    fft = np.fft.fft2( img_np, axes=(-2, -1) )
    return np.abs(fft)

def extractor_savor(dir_path, list_of_data_path, client_idx, split='train'):
    for data_idx, each_data_path in enumerate(list_of_data_path):
        img_path = each_data_path['image']
        label_path = each_data_path['label']
        img = Image.open(os.path.join(dir_path, img_path))
        label = Image.open(os.path.join(dir_path, label_path))
        img = img.resize((256,256), Image.BICUBIC )
        label = label.resize((256,256), Image.NEAREST )
        img_np = np.asarray(img)
        amp = extract_amp_spectrum(img_np)
        label_np = np.expand_dims(np.asarray(label)/255., axis=-1)
        # concatenate channel dimension
        img_np = np.concatenate([img_np, label_np ], axis=-1)
        #print(img_np.shape, img_np.dtype, img_np.max(), img_np.min())
        #print(label_np.shape, label_np.max(), label_np.min())

        save_dir_data = os.path.join(dir_path, "Data_FEDDG", f"dataset_{split}", f'client{client_idx}', 'data_npy' )
        save_dir_amp = os.path.join(dir_path, "Data_FEDDG", f"dataset_{split}", f'client{client_idx}', 'freq_amp_npy' )
        if not os.path.exists(save_dir_data):
            os.makedirs(save_dir_data)
        np.save(os.path.join(save_dir_data, f'sample{data_idx}.npy'), img_np)

        
        if not os.path.exists(save_dir_amp):
            os.makedirs(save_dir_amp)
        np.save(os.path.join(save_dir_amp, f'amp_sample{data_idx}.npy'), amp)

using_dataset_sites = [1, 2, 3, 4, 6]
dir_path = "/media/abhijeet/Data/Retinal/Extract/"
for new_client, og_client in enumerate(using_dataset_sites):
    data_path = f"{dir_path}/dataset_site{og_client}.json"
    data = read_json(data_path)
    extractor_savor(dir_path, data['training'], new_client+1, split='train')
    extractor_savor(dir_path, data['validation'], new_client+1, split='val')

    
