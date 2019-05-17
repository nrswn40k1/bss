"""
新しいグループ用のディレクトリをDropbox/group/, video/group/ に作成する
"""

import os


def dircount(path):
    dirlist = []
    for dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, dir)):
            dirlist.append(dir)
    return len(dirlist)


def main():
    grouppath = "./Dropbox/group"
    ngroup = dircount(grouppath)
    nthgroup = "group{}".format(ngroup)
    dirname = os.path.join(grouppath, nthgroup)

    assert not os.path.isdir(dirname), "{} already exists in {}".format(nthgroup, grouppath)

    os.mkdir(dirname)
    for i in range(3):
        os.mkdir(os.path.join(dirname, "speaker{}".format(i)))
    os.mkdir(os.path.join(dirname, "target"))

    videopath = "./Video/group"
    dirname = os.path.join(videopath, nthgroup)

    assert not os.path.isdir(dirname), "{} already exists in {}".format(nthgroup, videopath)

    os.mkdir(dirname)


if __name__ == "__main__":
    main()
