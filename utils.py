import time
import datetime


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        diff_time = time2-time1
        if diff_time < 5:
            print('%s function took %0.3f ms' % (f.__name__, diff_time*1000.0))
        else:
            print('%s function took %0.3f s' % (f.__name__, diff_time))
        return ret
    return wrap


def get_file_friendly_datetime_string():
    return datetime.datetime.now().strftime('%Y.%m.%d - %X').replace(':', '.')


def merge_csv_files(csv_filename_list, merge_filename):
    """ Merges multiple csv files into one, assumes there is a header row. """
    file_out = open(merge_filename, "wt", encoding='utf-8')
    try:
        # first file:
        for line in open(csv_filename_list[0], "rt", encoding='utf-8'):
            file_out.write(line)
        # now the rest:
        for num in range(1, len(csv_filename_list)):
            f = open(csv_filename_list[num], "rt", encoding='utf-8')
            f.readline()  # skip the header
            for line in f:
                file_out.write(line)
            f.close()  # not really needed
    except ValueError as ve:
        print(ve)
    finally:
        file_out.close()
