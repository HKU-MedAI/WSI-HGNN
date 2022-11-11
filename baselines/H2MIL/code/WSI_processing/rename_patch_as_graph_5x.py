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
        # print(wsi_patch_list)
        # import sys
        # sys.exit()
        # rename 5x patch
        count = 0
        for low_mag_patch in wsi_patch_list:
            # if count == 12:
            #     print('-1' not in os.path.basename(low_mag_patch))
            #     import sys
            #     sys.exit()
            if low_mag_patch.endswith('.jpeg') and '-1' not in os.path.basename(low_mag_patch) and '-' not in (low_mag_patch.split('/')[-1]):
                low_mag_coordinate = low_mag_patch.split('/')[-1].split('.')[0]
                sub_patch_folder = low_mag_patch[0:-5]
                sub_patch_list = glob(sub_patch_folder + '/*.jpeg')
                sub_x = list()
                sub_y = list()
                for sub_patch in sub_patch_list:
                    sub_x.append(int(sub_patch.split('/')[-1].split('.')[0].split('_')[0]))
                    sub_y.append(int(sub_patch.split('/')[-1].split('.')[0].split('_')[1]))
                sub_x = list(set(sub_x))
                sub_y = list(set(sub_y))
                if len(sub_x) == 1:
                    sub_x.append(sub_x[0])
                if len(sub_y) == 1:
                    sub_y.append(sub_y[0])

                if len(sub_x) == 0 or len(sub_y) == 0:
                    os.remove(low_mag_patch)
                    continue

                sub_x.sort()
                sub_y.sort()
                x = int(low_mag_coordinate.split('_')[0])
                y = int(low_mag_coordinate.split('_')[1])
                try:
                    new_name = str(sub_x[0]) + '-' + str(sub_x[1]) + '-' + str(sub_y[0]) + '-' + str(sub_y[1]) + '-' + str(x) + '-' + str(y)
                    new_path = low_mag_patch[0 : -1 * len(low_mag_patch.split('/')[-1])] + new_name + '.jpeg'

                    shutil.move(low_mag_patch, new_path)
                    print("move: " + low_mag_patch + "to: " +new_path)
                except IndexError:
                    print("a")
            # count +=1

        time_elapsed = time.time() - since
        print('Evaluation completed in {:.0f}m {:.0f}s for wsi {:d}'.format(time_elapsed // 60, time_elapsed % 60,                                                                            wsi_counter))
        wsi_counter = wsi_counter + 1
