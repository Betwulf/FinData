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
