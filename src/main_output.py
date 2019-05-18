"""
最終的な動画の作成
"""

import os
from FacialRecog import FacialRecog
from speaker_recognition import speaker_recognition
import numpy as np
import cis


def dircount(path):
    dirlist = []
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            dirlist.append(dir)
    return len(dirlist)


def main():
    n_people = 3
    grouppath = "./Dropbox/group/"
    ngroup = dircount(grouppath)

    SR = speaker_recognition(n_people, os.path.join(grouppath, "group{}".format(ngroup-1)))
    index = SR.transform()

    fname = os.path.join(grouppath, "group{}/target/ilrma_{}.wav".format(ngroup - 1, int(index[0])))

    afps, data = cis.wavread(fname)
    x = np.array(data)
    for i in range(1, n_people):
        fname = os.path.join(grouppath, "group{}/target/ilrma_{}.wav".format(ngroup - 1, int(index[1])))
        _, data = cis.wavread(fname)
        x = np.vstack([x, data.astype(float)])

    fname = os.path.join(grouppath, "group{}/target/input.wav".format(ngroup - 1))
    _, data = cis.wavread(fname)

    sepvoice = x
    rawvoice = np.array(data, dtype=np.float32)

    vfname = "./Video/group/group{}/input.avi".format(ngroup-1)
    outfile = "./Video/group/group{}/output".format(ngroup-1)
    recog = FacialRecog(vfname, rawvoice[:, 0], sepvoice, afps, outfile)
    recog.main()


if __name__ == '__main__':
    main()
