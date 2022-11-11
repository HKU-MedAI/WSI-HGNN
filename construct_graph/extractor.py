import time
import cv2

import numpy as np
import torch

from openslide import OpenSlide, OpenSlideUnsupportedFormatError

from globals import PIXEL_WHITE


class Extractor:
    def __init__(self, config, wsi_data):
        # Categorize configs
        self.config = config

        # Load file path
        self.tif_file_path = wsi_data

        # Load levels, patch size and verbose
        self.level = self.config['level']
        self.mag_factor = pow(2, self.level)
        self.patch_size = self.config['patch_size']
        verbose = self.config['verbose']
        self.verboseprint = print if verbose else lambda *a, **k: None

        # Load channel
        self.n_channel = self.config['n_channel']

    def extract_patches(self):
        """
        ref 1: https://github.com/tcxxxx/WSI-analysis/tree/master/patch_extraction
        ref 2: https://developer.ibm.com/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/
        :param slide: The given slides to be segmented
        :return: regions of interests
        """
        ####################
        # Step 1 Segment the image
        ####################
        wsi_, rgba_, shape_ = self.read_wsi()
        
        wsi_rgb_, wsi_gray_, wsi_hsv_ = self.construct_colored_wsi(rgba_)
        bounding_boxes, contour_coords, contours, mask = self.segmentation_hsv(wsi_hsv_, wsi_rgb_)
        ####################
        # Step 2 Extract ROIs from regions
        ####################
        patches, patches_coords = self.construct_bags(wsi_, wsi_rgb_, contours, mask)

        # Permute the dimensions and to torch tensor
        patches = [torch.from_numpy(np.moveaxis(p, -1, 0)).type(torch.FloatTensor) for p in patches]

        # Convert coordinates to numpy array
        patches_coords = np.array(patches_coords)

        return patches, patches_coords, mask

    def read_wsi(self):
        """
            Identify and load slides.
            Returns:
                - wsi_image: OpenSlide object.
                - rgba_image: WSI image loaded, NumPy array type.
        """

        time_s = time.time()

        try:
            wsi_image = OpenSlide(self.tif_file_path)
            slide_w_, slide_h_ = wsi_image.level_dimensions[self.level]

            """
                The read_region loads the target area into RAM memory, and
                returns an Pillow Image object.
                !! Take care because WSIs are gigapixel images, which are could be 
                extremely large to RAMs.
                Load the whole image in level < 3 could cause failures.
            """

            # Here we load the whole image from (0, 0), so transformation of coordinates
            # is not skipped.

            rgba_image_pil = wsi_image.read_region((0, 0), self.level, (slide_w_, slide_h_))
            self.verboseprint("width, height:", rgba_image_pil.size)

            """
                !!! It should be noted that:
                1. np.asarray() / np.array() would switch the position 
                of WIDTH and HEIGHT in shape.
                Here, the shape of $rgb_image_pil is: (WIDTH, HEIGHT, CHANNEL).
                After the np.asarray() transformation, the shape of $rgb_image is: 
                (HEIGHT, WIDTH, CHANNEL).
                2. The image here is RGBA image, in which A stands for Alpha channel.
                The A channel is unnecessary for now and could be dropped.
            """
            rgba_image = np.asarray(rgba_image_pil)
            self.verboseprint("transformed:", rgba_image.shape)

        except OpenSlideUnsupportedFormatError:
            print('Exception: OpenSlideUnsupportedFormatError')
            return None

        time_e = time.time()

        self.verboseprint("Time spent on loading", self.tif_file_path, ": ", (time_e - time_s))

        return wsi_image, rgba_image, (slide_w_, slide_h_)

    @staticmethod
    def construct_colored_wsi(rgba_):
        """
            This function splits and merges R, G, B channels.
            HSV and GRAY images are also created for future segmentation procedure.
            Args:
                - rgba_: Image to be processed, NumPy array type.
        """
        r_, g_, b_, a_ = cv2.split(rgba_)

        wsi_rgb_ = cv2.merge((r_, g_, b_))

        wsi_gray_ = cv2.cvtColor(wsi_rgb_, cv2.COLOR_RGB2GRAY)
        wsi_hsv_ = cv2.cvtColor(wsi_rgb_, cv2.COLOR_RGB2HSV)

        return wsi_rgb_, wsi_gray_, wsi_hsv_

    def segmentation_hsv(self, wsi_hsv_, wsi_rgb_):
        """
        This func is designed to remove background of WSIs.
        Args:
            - wsi_hsv_: HSV images.
            - wsi_rgb_: RGB images.
        Returns:
            - bounding_boxs: List of regions, region: (x, y, w, h);
            - contour_coords: List of arrays. Each array stands for a valid region and
            contains contour coordinates of that region.
            - contours: Almost same to $contour_coords;
            - mask: binary mask array;
            !!! It should be noticed that:
            1. The shape of mask array is: (HEIGHT, WIDTH, CHANNEL);
            2. $contours is unprocessed format of contour list returned by OpenCV cv2.findContours method.

            The shape of arrays in $contours is: (NUMBER_OF_COORDS, 1, 2), 2 stands for x, y;
            The shape of arrays in $contour_coords is: (NUMBER_OF_COORDS, 2), 2 stands for x, y;
            The only difference between $contours and $contour_coords is in shape.
        """
        self.verboseprint("HSV segmentation: ")
        contour_coord = []

        '''
            Here we could tune for better results.
            Currently 20 and 200 are lower and upper threshold for H, S, V values, respectively. 

            !!! It should be noted that the threshold values here highly depends on the dataset itself.
            Thresh value could vary a lot among different datasets.
        '''
        lower_ = np.array([20, 20, 20])
        upper_ = np.array([200, 200, 200])

        # HSV image threshold
        thresh = cv2.inRange(wsi_hsv_, lower_, upper_)

        try:
            self.verboseprint("thresh shape:", thresh.shape)
        except:
            self.verboseprint("thresh shape:", thresh.size)
        else:
            pass

        '''
            Closing
        '''
        self.verboseprint("Closing step: ")
        close_kernel = np.ones((15, 15), dtype=np.uint8)
        image_close = cv2.morphologyEx(np.array(thresh), cv2.MORPH_CLOSE, close_kernel)
        self.verboseprint("image_close size", image_close.shape)

        '''
            Openning
        '''
        self.verboseprint("Openning step: ")
        open_kernel = np.ones((5, 5), dtype=np.uint8)
        image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, open_kernel)
        self.verboseprint("image_open size", image_open.size)

        self.verboseprint("Getting Contour: ")
        bounding_boxes, contour_coords, contours, mask \
            = self.get_contours(np.array(image_open), wsi_rgb_.shape)

        return bounding_boxes, contour_coords, contours, mask

    def get_contours(self, cont_img, rgb_image_shape):
        """
        Args:
            - cont_img: images with contours, these images are in np.array format.
            - rgb_image_shape: shape of rgb image, (HEIGHT, WIDTH).
        Returns:
            - bounding_boxs: List of regions, region: (x, y, w, h);
            - contour_coords: List of valid region coordinates (contours squeezed);
            - contours: List of valid regions (coordinates);
            - mask: binary mask array;
            !!! It should be noticed that the shape of mask array is: (HEIGHT, WIDTH, CHANNEL).
        """

        self.verboseprint('contour image: ', cont_img.shape)

        contour_coords = []
        contours, hiers = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        # print(contours)
        boundingBoxes = [cv2.boundingRect(c) for c in contours]

        for contour in contours:
            contour_coords.append(np.squeeze(contour))

        mask = np.zeros(rgb_image_shape, np.uint8)

        self.verboseprint('mask shape', mask.shape)
        cv2.drawContours(mask, contours, -1, (PIXEL_WHITE, PIXEL_WHITE, PIXEL_WHITE), thickness=-1)

        return boundingBoxes, contour_coords, contours, mask

    def construct_bags(self, wsi_, wsi_rgb, contours, mask):
        """
        Args:
            To-do.
        Returns:
            - patches: lists of patches in numpy array: [PATCH_WIDTH, PATCH_HEIGHT, CHANNEL]
            - patches_coords: coordinates of patches: (x_min, y_min). The bouding box of the patch
            is (x_min, y_min, x_min + PATCH_WIDTH, y_min + PATCH_HEIGHT)
        """

        patches = []
        patches_coords = []

        start = time.time()

        '''
            !!! 
            Currently we select only the first 5 regions, because there are too many small areas and 
            too many irrelevant would be selected if we extract patches from all regions.
            And how many regions from which we decide to extract patches is 
            highly related to the SEGMENTATION results.
        '''
        contours_ = sorted(contours, key=cv2.contourArea, reverse=True)
        contours_ = contours_[:5]

        for i, box_ in enumerate(contours_):

            box_ = cv2.boundingRect(np.squeeze(box_))
            self.verboseprint('region', i)

            '''
            !!! Take care of difference in shapes:
                Coordinates in bounding boxes: (WIDTH, HEIGHT)
                WSI image: (HEIGHT, WIDTH, CHANNEL)
                Mask: (HEIGHT, WIDTH, CHANNEL)
            '''

            b_x_start = int(box_[0])
            b_y_start = int(box_[1])
            b_x_end = int(box_[0]) + int(box_[2])
            b_y_end = int(box_[1]) + int(box_[3])

            '''
                !!!
                step size could be tuned for better results.
            '''

            X = np.arange(b_x_start, b_x_end, step=self.patch_size // 2)
            Y = np.arange(b_y_start, b_y_end, step=self.patch_size // 2)

            self.verboseprint('ROI length:', len(X), len(Y))

            for h_pos, y_height_ in enumerate(Y):

                for w_pos, x_width_ in enumerate(X):

                    # Read again from WSI object wastes tooooo much time.
                    # patch_img = wsi_.read_region((x_width_, y_height_), level, (self.patch_size, self.patch_size))

                    '''
                        !!! Take care of difference in shapes
                        Here, the shape of wsi_rgb is (HEIGHT, WIDTH, CHANNEL)
                        the shape of mask is (HEIGHT, WIDTH, CHANNEL)
                    '''
                    patch_arr = wsi_rgb[y_height_: y_height_ + self.patch_size, \
                                x_width_:x_width_ + self.patch_size, :]
                    self.verboseprint("read_region (scaled coordinates): ", x_width_, y_height_)

                    width_mask = x_width_
                    height_mask = y_height_

                    patch_mask_arr = mask[height_mask: height_mask + self.patch_size, \
                                     width_mask: width_mask + self.patch_size]

                    self.verboseprint("Numpy mask shape: ", patch_mask_arr.shape)
                    self.verboseprint("Numpy patch shape: ", patch_arr.shape)

                    try:
                        bitwise_ = cv2.bitwise_and(patch_arr, patch_mask_arr)

                    except Exception as err:
                        print('Out of the boundary')
                        pass

                    #                     f_ = ((patch_arr > PIXEL_TH) * 1)
                    #                     f_ = (f_ * PIXEL_WHITE).astype('uint8')

                    #                     if np.mean(f_) <= (PIXEL_TH + 40):
                    #                         patches.append(patch_arr)
                    #                         patches_coords.append((x_width_, y_height_))
                    #                         print(x_width_, y_height_)
                    #                         print('Saved\n')

                    else:
                        bitwise_grey = cv2.cvtColor(bitwise_, cv2.COLOR_RGB2GRAY)
                        white_pixel_cnt = cv2.countNonZero(bitwise_grey)

                        '''
                            Patches whose valid area >= 25% of total area is considered
                            valid and selected.
                        '''

                        if white_pixel_cnt >= ((self.patch_size ** 2) * 0.25):

                            if patch_arr.shape == (self.patch_size, self.patch_size, self.n_channel):
                                patches.append(patch_arr)
                                patches_coords.append((x_width_, y_height_))
                                self.verboseprint(x_width_, y_height_)
                                self.verboseprint('Saved\n')

                        else:
                            self.verboseprint('Did not save\n')

        end = time.time()
        self.verboseprint("Time spent on patch extraction: ", (end - start))

        # patches_ = [patch_[:,:,:3] for patch_ in patches]
        print("Total number of patches extracted:", len(patches))

        return patches, patches_coords
