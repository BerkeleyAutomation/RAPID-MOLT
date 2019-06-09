'''
This is the main script executing the image processing of the raw RGB and thermal images.

The project folder has the following structure:
Project folder
    0_Data_raw
        2019_04_05_experiment
            RGB_image_123
            Thermal_image_123
    1_Data_processed
        <resulting csv and avi files>

The path to the folder should be adapted. Also the ranges for the image processing
the HSV ranges for the background and leafs have to be individually adapted.

Output:
    *.csv file of the LAI and leaf temperature data across time
    [if debug_video = true] an *.avi file over time
'''

import datetime
import numpy as np
import sys
from matplotlib import pyplot as plt
import time
import os
import argparse
import cv2

from data_processing_utils import *
from image_processing_utils import *
import csv
from tqdm import tqdm


def main(args):

    # Setup
    # Folder for the project data
    project_path = '/Graduate_Research/RAPID_MOLT_Project/'
    csv_file = open(project_path + '1_Data_processed/' + args.data_folder + '.csv', 'w')    # to store results in
    file_path = project_path + '0_Data_raw/' + args.data_folder + '/'   # path to raw data

    # Ranges of the background to mask out (for us it was orange)
    mask_out_hue_ranges = [(0, 37), (173, 180)]
    mask_out_sat_range = [24, 255]
    # values to select only the leafs (H, S, V)
    lower_green = np.array([45, 100, 80])
    upper_green = np.array([85, 255, 255])
    hsv_green_range = [lower_green, upper_green]

    # initialize csv file
    lai_numbered = ["LAI_pot_" + str(i) for i in range(20)]
    temp_numbered = ["temp_" + str(i) for i in range(20)]

    # set the WARS pot
    lai_numbered[args.wet_pot_nr] = 'wet_LAI'
    temp_numbered[args.wet_pot_nr] = 'wet_temp'
    csv_columns = ['Timestamp', 'year', 'month', 'day', 'hour', 'minute', 'second'] + lai_numbered + temp_numbered + ['dry_temp']
    wr = csv.writer(csv_file, dialect='excel')
    wr.writerow(csv_columns)

    # initialize the loop (logging how many images are not good)
    n_bad = 0

    # get RGB and thermal image file list, and sort it
    file_list = os.listdir(file_path)
    sorted_list = sort_by_time(file_list)

    # if fixed contour selected: calculate fixed pot contours from the first image
    if args.fixed_contours or args.fixed_leaf_mask:
        # get one green mask for all pictures
        next_pic = True
        i = 0
        while next_pic:
            ref_rgb_file = sorted_list[i][1]
            ref_thermal_file = sorted_list[i][2]
            ref_rgb_im = cv2.imread(file_path + ref_rgb_file)  # numpy array 640x480x3
            ref_therm_im = cv2.imread(file_path + ref_thermal_file)  # numpy array 320x240
            # check if bad image, then go to next
            _, _, BDP = cv2.split(ref_therm_im)
            check = check_image(BDP, ref_rgb_file, sorted_list[i][1])
            if not check:
                i += 1
                continue
            cnts, next_pic = find_pot_centers(ref_rgb_im, mask_out_hue_ranges, mask_out_sat_range, debug=True)
            i += 1
        #  if fixed leaf masks selected: calculate the fixed leaf masks
        if args.fixed_leaf_mask:
            # Leaf mask and LAI analyzer
            leaf_pot_masks, rgb_masked, rgb_masked_wg = LAI_and_leaf_masks(cnts, ref_rgb_im, args.wet_pot_nr, csv_file, hsv_green_range, debug=True)

    # initialize video out
    (h, w, c) = cv2.imread(file_path + sorted_list[i][1]).shape
    vid_rgb = cv2.VideoWriter(project_path + '1_Data_processed/' + args.data_folder + '_rgb.avi',
                              cv2.VideoWriter_fourcc(*'DIVX'), 10, (2*w, h+60))
    vid_therm = cv2.VideoWriter(project_path + '1_Data_processed/' + args.data_folder + '_therm.avi',
                              cv2.VideoWriter_fourcc(*'DIVX'), 10, (2 * w, h + 60))
    if args.debug_vid:
        debug_vid_writer = cv2.VideoWriter(project_path + '1_Data_processed/' + args.data_folder + '_debug_vid.avi',
                                           cv2.VideoWriter_fourcc(*'DIVX'), 10, (4*w, h+60))
    # process all images
    with tqdm(total=len(sorted_list)) as progress_bar:
        for idx, item in enumerate(sorted_list):
            # read in the image data
            rgb_im = cv2.imread(file_path + item[1])  # numpy array 640x480x3
            # check if thermal image exists
            if not os.path.exists(file_path + item[2]):
                continue
            thermal_im = cv2.imread(file_path + item[2])  # numpy array 320x240x3
            _, _, BDP = cv2.split(thermal_im)  # split the RGB values into the individual values for each pixel

            # sort out bad images (either night, or thermal camera bad reading)
            if not check_image(BDP, rgb_im, item[1]):
                n_bad += 1
                continue

            # Start a new line in the CSV file
            log_csv(csv_file, item)

            # calculate the contours for each image individually (only needed if camera was moved)
            if not (args.fixed_contours or args.fixed_leaf_mask):
                # returns a list of the center points of pots and a mask of them
                cnts, _ = find_pot_centers(rgb_im, mask_out_hue_ranges, mask_out_sat_range, debug=False)
            if not args.fixed_leaf_mask:  # calculate the green masks for each picture individually
                # Leaf mask and LAI analyzer
                leaf_pot_masks, rgb_masked, rgb_masked_wg = LAI_and_leaf_masks(cnts, rgb_im, args.wet_pot_nr,
                                                                               csv_file, hsv_green_range, debug=False)

            # T canopy and T dry analyzer
            thermal_rgb, therm_masked = thermal_image_analyze(cnts, thermal_im, leaf_pot_masks, csv_file,
                                                              median=True)
            # update video stream
            update_vid(np.concatenate((rgb_im, rgb_masked), axis=1), item, vid_rgb)
            update_vid(np.concatenate((thermal_rgb, therm_masked), axis=1), item, vid_therm)
            if args.debug_vid:
                update_vid(np.concatenate((rgb_im, rgb_masked, thermal_rgb, therm_masked), axis=1), item,
                           debug_vid_writer)

            # update the progress bar
            progress_bar.set_postfix(curr=item[1])
            progress_bar.update(100*idx/len(sorted_list))

    print('\n')
    print("Percentage of images discarded: ", 100 * n_bad / len(sorted_list))

    cv2.destroyAllWindows()
    vid_rgb.release()
    vid_therm.release()
    debug_vid_writer.release()


def log_csv(csv_file, item):
    # starts a new line with timestamp etc in the csv file
    csv_file.write('\n')

    RGB_item = item[1]

    t = datetime.datetime(int(RGB_item[4:8]), int(RGB_item[9:11]), int(RGB_item[12:14]), int(RGB_item[15:17]),
                          int(RGB_item[18:20]), int(RGB_item[21:23]))
    t_diff = time.mktime(t.timetuple())
    csv_file.write(str(t_diff) + ',')
    csv_file.write(str(RGB_item[4:8]) + ',')
    csv_file.write(str(RGB_item[9:11]) + ',')
    csv_file.write(str(RGB_item[12:14]) + ',')
    csv_file.write(str(RGB_item[15:17]) + ',')
    csv_file.write(str(RGB_item[18:20]) + ',')
    csv_file.write(str(RGB_item[21:23]) + ',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for the image analysis')

    parser.add_argument('--fixed_leaf_mask', default=False, type=bool, help='leaf mask fixed or calculated for each image')
    parser.add_argument('--fixed_contours', default=True, help='fixed pot contours or calculate for each image')
    parser.add_argument('--wet_pot_nr', default=7, type=int, help='number of the pot with the wet cloth')
    parser.add_argument('--data_folder', default='2019_exp10',  type=str, help='name of the experiment-folder to analyse')
    parser.add_argument('--debug_vid', default=True, type=bool, help='creates a debug video')
    main(parser.parse_args())
