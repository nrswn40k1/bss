"""
MFCC のトレーニング用に初めに各参加者の声を録音する
"""

import record
from record import start_audio_recording, stop_AVrecording
import os
import time
import sys


def dircount(path):
    dirlist = []
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            dirlist.append(dir)
    return len(dirlist)


def main(person):
    grouppath = "./Dropbox/group"
    ngroup = dircount(grouppath)
    dirname = os.path.join(grouppath, "group{}/speaker{}/".format(ngroup - 1, person))

    print("recording...")
    start_audio_recording(1, dirname, "speaker{}.wav".format(person))
    time.sleep(11)
    record.audio_thread.stop()
    print("end")


if __name__ == '__main__':

    person = sys.argv[1]
    main(person)
