"""
実演でILRMA を実行するコード
"""

import os
import numpy as np

from AuxIVA import AuxIVA
from main_preparation import dircount
import scipy.io.wavfile as wf


def main():
    grouppath = "./Dropbox/group"
    ngroup = dircount(grouppath)
    dirname = os.path.join(grouppath, "group{}/target".format(ngroup-1))
    """
    fs, data = cis.wavread(os.path.join(dirname, "input.wav"))
    x = np.array([data[:, 0], data[:, 1], data[:, 2]], dtype=np.float32)
    """

    rate0, data0 = wf.read(os.path.join(dirname, "input.wav"))

    data0 = data0.astype(float).T

    y = AuxIVA(data0, sample_freq=rate0, beta=0.3).auxiva()

    y = [(y_i * 32767 * 2/ max(np.absolute(y_i))).astype(np.int16) for y_i in np.asarray(y)]


    wf.write(os.path.join(dirname, "auxiva_0.wav"), rate0, y[0])
    wf.write(os.path.join(dirname, "auxiva_1.wav"), rate0, y[1])
    wf.write(os.path.join(dirname, "auxiva_2.wav"), rate0, y[2])

    """
    cis.wavwrite(os.path.join(dirname, "auxiva_0.wav"), fs, y[0])
    cis.wavwrite(os.path.join(dirname, "auxiva_1.wav"), fs, y[1])
    cis.wavwrite(os.path.join(dirname, "auxiva_2.wav"), fs, y[2])
    """


if __name__ == "__main__":
    main()
