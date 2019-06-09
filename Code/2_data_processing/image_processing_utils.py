import cv2
import numpy as np


def find_pot_centers(img, mask_out_hue_ranges, mask_out_sat_range, debug=False):
    """
    takes an RGB image, masks out specific hue ranges (the background) to find contours
    and then returns the sorted list of pot centers pixels.
    """
    # split into HSV
    hsv_im = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_im)

    th = np.zeros(h.shape).astype(np.uint8)
    for hue_range in mask_out_hue_ranges:
        th_new = cv2.inRange(h, hue_range[0], hue_range[1]) # sets all pixels to 255 where h is in range otherwise 0
        th_new = (cv2.bitwise_not(th_new) > 1).astype(np.uint8) # puts a 1 where the pixel was not in range
        th = th + th_new    # combine threshold with or's

    # mask out saturation range (white)
    sat_mask = (cv2.inRange(s, mask_out_sat_range[0], mask_out_sat_range[1]) > 1).astype(np.uint8)

    # make the aggregated or into a logical mask
    th = (cv2.multiply(th, sat_mask) > 1).astype(np.uint8)   # a logical operation, if either of the ranges AND white => true

    # cleaning threshold
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel1)

    cnts, _ = cv2.findContours(th.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # extraxt the center points for each of the contours
    cnts_new = np.zeros((len(cnts), 2))
    out_cnts = np.zeros((len(cnts), 2))
    for i in range(0, len(cnts), 1):
        m = cv2.moments(cnts[i])
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        cnts_new[i, 0] = cx
        cnts_new[i, 1] = cy

    # sort the centers according to their x-coordinate value
    I = np.argsort(cnts_new[:, 0])
    sorted_cnts = cnts_new[I, :]
    # print sorted_cnts
    for col in range(0, 4, 1):
        arr = sorted_cnts[col * 5:col * 5 + 5, :]
        I = np.argsort(arr[:, 1])      # sort in y axis
        I = I[::-1]     # array from back to front
        out_cnts[col * 5:col * 5 + 5, :] = arr[I, :]

    next_pic = False
    # visualization in debug mode
    if debug:

        cv2.imshow('Masked RGB', th.reshape((h.shape[0], h.shape[1], 1)) * img)
        draw_contour = img.copy()   # make a copy because otherwise overwrites the object in the memory
        img_conts = cv2.drawContours(draw_contour, cnts, -1, (0, 0, 255), 3)
        cv2.imshow('RGB_with_contours', img_conts)
        cv2.waitKey()
        next_pic = opencv_wait()

    # return the list of center pixels (ordered) and the mask
    return out_cnts, next_pic


def opencv_wait():
    if cv2.waitKey() == 32:  # space means accept the boundaries
        next_pic = False
        print("accept")
    else:
        next_pic = True
        print("next pic")
    return next_pic


def get_heatmap(thermal_image, res=(640, 480), relative=True, min_max=[18, 28]):
    """
    takes in the thermal image and returns an rgb heatmap of it
    """
    _, adp, bdp = cv2.split(thermal_image)  # bdp - Before Decimal Point, adp - After Decimal Point
    real_tmp_val_im = bdp.astype(np.float) + adp.astype(np.float) / 100     # this is a matrix 320 x 240

    # normalizing and presenting the real_tmp_val_im (only for visual)
    if relative:
        im_min = np.min(real_tmp_val_im).astype(float)
        blank_image = real_tmp_val_im - im_min
        im_max = np.max(blank_image).astype(float)
        blank_image = blank_image * 255 / im_max  # scale color value up to 255
    else:
        im_min = min_max[0]
        blank_image = real_tmp_val_im - im_min
        im_max = min_max[1]
        blank_image = blank_image * 255 / (im_max-im_min)  # scale color value up to 255

    thermal_im = cv2.applyColorMap(blank_image.astype(np.uint8), cv2.COLORMAP_JET)
    (h, w) = res
    thermal_im = cv2.resize(thermal_im, (w, h))

    return thermal_im


def LAI_and_leaf_masks(out_cnts, rgb_im, wet_pot, csv_file, hsv_green_range,  debug=False):
    """
    Calculates and writes the LAI for each pot to the csv and produces debug images.
    input:
        rgb_im              RGB image
        out_cnts            list of pot center pixels
        wet_pot             number of the WARS pot
        csv_file            csv file to write the LAI results to
        hsv_green_range     HSV range to mask for the leaves

    return:
        pot_green_masks     the {0,1} mask of the leave pixels
        rgb_masked_full     fully masked rgb (only leaves visible)
        rgb_masked_wg       intermediate rgb masks (only leaves)
    """
    circle_rad = 40   # depends on the size of the pots used and the distance to the camera

    # values for masking out the black pipes in the image
    low_non_black = np.array([0, 0, 120])
    high_non_black = np.array([179, 255, 255])
    lower_green = hsv_green_range[0]
    upper_green = hsv_green_range[1]

    (h, w, _) = rgb_im.shape

    # mask the green parts
    hsv = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)  # this mask is 0 or 255!
    # delete small parts
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

    b_mask = (mask > 0).astype(float)

    pot_green_masks = []        # sort in the individual pot masks
    rgb_masked_full = np.zeros(rgb_im.shape).astype(np.uint8)
    rgb_masked_wg = np.zeros(rgb_im.shape).astype(np.uint8)

    for n_pot, center in enumerate(out_cnts):
        cx = int(center[0])
        cy = int(center[1])

        pot_mask = np.zeros((h, w), np.uint8)
        cv2.circle(pot_mask, (cx, cy), circle_rad, 1, -1)   # create a circular mask
        if n_pot != wet_pot:
            pot_green_mask = b_mask * pot_mask
        else:   # pot is the wet pot
            mask_2 = (cv2.inRange(hsv, low_non_black, high_non_black) > 0).astype(float)
            pot_green_mask = pot_mask * mask_2
        pot_green_masks.append(pot_green_mask)

        # calculate the LAI approximation (by counting the green pixels)
        leaf_pixel_ratio = np.sum(pot_green_mask)/np.sum(pot_mask)

        # write it to the CSV
        csv_file.write(str(leaf_pixel_ratio) + ',')

        # just for debug purpose
        # rgb masked
        rgb_masked = np.uint8(rgb_im * pot_green_mask.reshape((h, w, 1)))
        cv2.circle(rgb_masked, (cx, cy), circle_rad, 255, 2)
        cv2.putText(rgb_masked, str(round(leaf_pixel_ratio, 2)), (cx - 5, cy + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rgb_masked_full += rgb_masked

        # pure rgb masked
        rgb_masked_wg_single = np.uint8(rgb_im * pot_mask.reshape((h, w, 1)))
        cv2.circle(rgb_masked_wg_single, (cx, cy), circle_rad, 255, 2)
        cv2.putText(rgb_masked_wg_single, str(round(leaf_pixel_ratio, 2)), (cx - 5, cy + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rgb_masked_wg += rgb_masked_wg_single

        if debug:
            cv2.imshow("mask", np.uint8(pot_green_mask * 255))
            cv2.imshow("raw_rgb_mask_function", rgb_im)
            cv2.imshow("green_mask", rgb_masked)
            cv2.waitKey()

    # add the LAI title to the images
    cv2.putText(rgb_masked_full, "Leaf Area Index", (180, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(rgb_masked_wg, "Leaf Area Index", (180, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if debug:
        cv2.imshow("full", rgb_masked_full)
        cv2.waitKey()

    return pot_green_masks, rgb_masked_full, rgb_masked_wg


def thermal_image_analyze(out_cnts, thermal_input, leaf_masks, csv_file, median=True):
    """
    Calculates and writes the median leaf temperature for each pot to the csv and produces debug images.
    input:
        thermal_input       RGB image
        out_cnts            list of pot center pixels
        leaf_masks          leaf masks from the LAI_and_leaf_masks() function
        csv_file            csv file to write the LAI results to

    return:
        thermal_rgb         full thermal rgb
        therm_masked_full   masked thermal rgb with median temperature on it
    """
    circle_rad = 40
    _, adp, bdp = cv2.split(thermal_input)  # bdp - Before Decimal Point, adp - After Decimal Point
    real_tmp_val_im = bdp.astype(np.float) + adp.astype(np.float) / 100     # this is a matrix 320 x 240

    # check shape of the thermal image
    h, w, _ = thermal_input.shape
    (h_rgb, w_rgb) = leaf_masks[0].shape

    # normalizing and presenting the real_tmp_val_im (only for visual, same size as RGB)
    thermal_rgb = get_heatmap(thermal_input, res=(h_rgb, w_rgb), relative=False)

    therm_masked_full = np.zeros(thermal_rgb.shape, np.uint8)
    pot_leaf_mask_all = np.zeros((h, w))

    for center, pot_leaf_mask in zip(out_cnts, leaf_masks):
        cx = int(center[0])
        cy = int(center[1])
        # thermal masked
        therm_masked = np.uint8(thermal_rgb * pot_leaf_mask.reshape((h_rgb, w_rgb, 1)))
        cv2.circle(therm_masked, (cx, cy), circle_rad, 255, 2)
        # downsample to thermal size
        pot_leaf_mask = cv2.resize(pot_leaf_mask, (w, h))
        real_temp_flat = (pot_leaf_mask * real_tmp_val_im).flatten()
        leaf_pixel_temps = real_temp_flat[np.flatnonzero(real_temp_flat)]
        if median:
            cv2.putText(therm_masked_full, "Median Temp", (180, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            to_csv = np.median(leaf_pixel_temps)
        else:
            cv2.putText(therm_masked_full, "Mean Temp", (180, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            to_csv = np.mean(leaf_pixel_temps)
        csv_file.write(str(to_csv) + ',')
        # print the value to the masked image
        cv2.putText(therm_masked, str(to_csv) + " C", (cx - 20, cy + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        therm_masked_full += therm_masked
        # for the dry temp later
        pot_leaf_mask_all += pot_leaf_mask

    # add dry temperature
    non_pot_temp_flat = (real_tmp_val_im * np.logical_not(pot_leaf_mask_all)).flatten()
    non_pot_temp_flat = non_pot_temp_flat[np.flatnonzero(non_pot_temp_flat)]    # delete the zeros in there
    to_csv = str(np.median(non_pot_temp_flat))
    csv_file.write(to_csv)

    # write out dry temp
    cv2.putText(therm_masked_full, "T_dry = " + str(to_csv) + " C", (180, 630),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return thermal_rgb, therm_masked_full


def check_image(t_im, rgb_im, rgb_item):
    """
    Checks if the image pair is reasonable (e.g. not taken at night and reasonable temperature range)
    input:
        t_im                raw thermal image
        rgb_im              raw rgb image
        rgb_item            name of the rgb image

    return:
        image_status        {0,1} if the image is reasonable or not
    """
    # if the average temperature of the image is larger than 35deg, the image considered to be false
    image_status = True

    # check for thermal image (average above 30 or variance above 7)
    if (np.average(t_im) > 30) or np.var(t_im) > 7:
        image_status = False
        # print("average temperature above 35")

    # check that the RGB image is not black (night time)
    if np.average(rgb_im) < 25:
        # print("RGB image black")
        image_status = False

    # check to ignore images capture between midnight and 7:30 am
    hour = rgb_item[15:17]
    minute = rgb_item[18:20]
    if int(hour) < 7 or (int(hour) == 7 and int(minute) < 30):
        image_status = False

    # # check if the color image is bad (high green values)
    # hsv = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    # n, _, _ = plt.hist(h.ravel(), 2, [0, 180])
    # per = round(float(n[1]) / float(n[0]) * 100, 2)
    # # print(per)
    # # plt.show(block=False)
    # # cv2.imshow('rgb', im)
    # # cv2.waitKey()
    # # plt.clf()
    # # cv2.destroyAllWindows()
    # # plt.close()
    # # hist = cv2.calcHist([h], [0], None, [180], [0, 180])
    # if per < 11:
    #     image_status = False
    #     print("high green values")
    return image_status


def update_vid(image, item, vid_out_writer):
    ''' updates the video write out '''
    time_text = item[1]

    (h, b, c) = image.shape
    final_img = np.concatenate((np.zeros((60, b, 3), np.uint8), image), axis=0)
    cv2.putText(final_img,
                'Date: ' + time_text[4:14] + '  Time: ' + time_text[15:17] + ':' + time_text[18:20],
                (40, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1, cv2.LINE_8)
    # cv2.imshow("full", final_img)
    # cv2.waitKey()

    vid_out_writer.write(final_img)