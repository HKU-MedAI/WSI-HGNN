from glob import glob
import time
import openslide
from PIL import Image

size=(1000, 1000)
dataset_path = ''
true_dataset_path = ''
scale = "pyramid"

dataset_folder_list = glob(dataset_path + '/*')
for folder in dataset_folder_list:
    if folder.endswith('pyramid') or folder.endswith('single') or folder.endswith('pyramid_0_1_thumbnail'):
        dataset_folder_list.remove(folder)

since = time.time()
wsi_counter = 0

for dataset_folder in dataset_folder_list:
    wsi_list = glob(dataset_folder+'/*.svs')
    for wsi in wsi_list:
        try:
            slide = openslide.OpenSlide(wsi)
            thumbnail = slide.get_thumbnail(size=size)
            class_folder = wsi.split('/')[-2]
            wsi_id = wsi.split('/')[-1].split('.')[0]
            save_path = true_dataset_path + '/' + scale + '/' + class_folder + '/' + wsi_id + '/-1.jpeg'

            new_image = Image.new('RGB', size, (255,255,255))
            new_image.paste(thumbnail, ((size[0]-thumbnail.size[0])//2,(size[1]-thumbnail.size[1])//2))

            new_image.save(save_path)

            time_elapsed = time.time() - since
            print('Evaluation completed in {:.0f}m {:.0f}s for wsi {:d}'.format(time_elapsed // 60, time_elapsed % 60,
                                                                            wsi_counter))
            wsi_counter = wsi_counter + 1

        except:
            print("failed for wsi: " + str(wsi))
