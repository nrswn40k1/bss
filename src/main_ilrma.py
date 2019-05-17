"""
実演でILRMA を実行するコード
"""

import os
import numpy as np

from ILRMA import ILRMA
from main_preparation import dircount
import cis


def main():
    grouppath = "./Dropbox/group"
    ngroup = dircount(grouppath)
    dirname = os.path.join(grouppath, "group{}/target".format(ngroup-1))

    fs, data = cis.wavread(os.path.join(dirname, "input.wav"))
    x = np.array([data[:, 0], data[:, 1], data[:, 2]], dtype=np.float32)

    y = ILRMA(x, fs, 2, 100).ilrma()

    cis.wavwrite(os.path.join(dirname, "ilrma_0.wav"), fs, y[0])
    cis.wavwrite(os.path.join(dirname, "ilrma_1.wav"), fs, y[1])
    cis.wavwrite(os.path.join(dirname, "ilrma_2.wav"), fs, y[2])


if __name__ == "__main__":
    main()
