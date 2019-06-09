import datetime
import time


def sort_by_time(file_list):
    '''
    Sorts the image files by time.
    :param file_list:
    :return: list of image names sorted by time
    '''

    imagesByDate = []
    for item in file_list:
        if item[0:3] == 'RGB':
            year = item[4:8]
            month = item[9:11]
            day = item[12:14]
            hour = item[15:17]
            minute = item[18:20]
            second = item[21:23]
            date_num = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            time_num = time.mktime(date_num.timetuple())
            obj = [time_num, item,
                   'Thermal_' + year + '-' + month + '-' + day + '-' + hour + '-' + minute + '-' + second + '-0700.png']
            imagesByDate.append(obj)
    imagesByDate.sort()
    return imagesByDate