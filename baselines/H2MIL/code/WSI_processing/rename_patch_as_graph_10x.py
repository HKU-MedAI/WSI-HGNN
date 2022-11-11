import os
import shutil
from glob import glob
import time
import openslide

patch_path = 'data/BRCA/pyramid'

since = time.time()
wsi_counter = 0

class_folder_list = glob(patch_path + '/*')
for class_folder in class_folder_list:
    wsi_folder_list = glob(class_folder + '/*')
    for wsi in wsi_folder_list:
        wsi_patch_list = glob(wsi + '/*')
        # rename the 10x patch and move it
        for high_mag_patch_folder in wsi_patch_list:
            if (not high_mag_patch_folder.endswith('.jpeg')) and ('-1' not in os.path.basename(high_mag_patch_folder)) and (
                    '-' not in (high_mag_patch_folder.split('/')[-1])):
                high_mag_patch_list = glob(high_mag_patch_folder + '/*')
                for high_mag_patch in high_mag_patch_list:
                    patch_name = high_mag_patch.split('/')[-1]
                    patch_name = patch_name.replace("_", "-")
                    new_path = wsi + '/' + patch_name
                    shutil.move(high_mag_patch, new_path)
                shutil.rmtree(high_mag_patch_folder)
                # print("a")
                # print("move: " + low_mag_patch + "to: " + new_path)
                # print("")
