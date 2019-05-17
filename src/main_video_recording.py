"""
videoと音声の同時レコーティング
"""

import record
from record import start_AVrecording, stop_AVrecording
import os
import time

def dircount(path):
    dirlist = []
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            dirlist.append(dir)
    return len(dirlist)


def main():
    grouppath = "./Dropbox/group"
    ngroup = dircount(grouppath)
    dirname = os.path.join(grouppath, "group{}/target/".format(ngroup - 1))

    print("recording...")
    start_AVrecording(1, './Video/group/group{}/'.format(ngroup - 1), 'input.avi', 3, dirname, 'input.wav')
    time.sleep(11)
    stop_AVrecording()
    print("end")


if __name__ == '__main__':
    main()
