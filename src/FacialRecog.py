import cv2
import numpy as np
import moviepy.editor as mp
from moviepy.audio.AudioClip import AudioClip
import cis

face_cascade_path = "./haarcascade/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)


class FacialRecog:

    def __init__(self, mvfile, rawvoice, sepvoice, afps, outfile="output"):
        """
        :param mvfile: path of the movie file
        :param voice:  2 dementional array whose x is source index and y is time index
        """
        self.mv = mvfile
        self.cap = cv2.VideoCapture(self.mv)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap.isOpened() else 1
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap.isOpened() else 1
        self.nframe = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap.isOpened() else 1
        self.vfps = int(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap.isOpened() else 1
        self.rawvoice = rawvoice
        self.sepvoice = sepvoice
        self.afps = afps
        self.M = sepvoice.shape[0]
        self.outfile = outfile

    def main(self):
        self.edit()
        self.set_audio()

    def edit(self):
        if self.cap.isOpened():
            """output"""
            fourcc = cv2.VideoWriter_fourcc(*'DIV3')
            writer = cv2.VideoWriter(self.outfile+".avi", fourcc, self.vfps, (self.width, self.height))

            """editing"""
            sft = self.nframe // (self.M+2)

            print("making video...")
            for i in range(self.nframe):
                ret, frame = self.cap.read()
                if not ret:
                    break
                elif i < sft or sft*(self.M+1) <= i:
                    writer.write(frame)
                    continue
                elif i % sft == 0:
                    index = i // sft - 1
                    bbox = self.__getbox(frame, index)
                    tracker = cv2.TrackerMIL_create()
                    _ = tracker.init(frame, tuple(bbox))
                else:
                    _, bbox = tracker.update(frame)
                outframe = self.__spotlight(frame, bbox)
                writer.write(outframe)
            writer.release()
        self.cap.release()

    def set_audio(self):
        print("set audio...")
        audio = self.__audio_concat()
        cis.wavwrite(self.outfile+".wav", self.afps, audio)
        videoclip = mp.VideoFileClip(self.outfile+".avi")
        videoclip.write_videofile(self.outfile+".mp4", audio=self.outfile+".wav",
                                  codec="h264")

    def __getbox(self, src, index):
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(src_gray)
        faces = sorted(faces, key=lambda x: x[0], reverse=True)

        return faces[index]

    def __spotlight(self, src, box):

        output = src

        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        fx1 = max(0, x-w//2)
        fx2 = min(x+w+w//2, src.shape[1])
        fy1 = max(0, y-h//2)
        fy2 = min(y+h+h//2, src.shape[0])

        output = output // 3
        output[fy1:fy2,fx1:fx2,:] *= 3

        return output

    def __audio_concat(self):
        sft = self.nframe // (self.M+2) * self.afps // self.vfps
        for m in range(self.M+2):
            if m < 1:
                output = self.rawvoice[sft*m:sft*(m+1)]
            elif m > self.M:
                output = np.hstack((output, self.rawvoice[sft*m:]))
            else:
                output = np.hstack((output, self.sepvoice[m-1, sft*m:sft*(m+1)]))
        return output


if __name__ == "__main__":
    rate1, data1 = cis.wavread('./samples/voice/ilrma_f1.wav')
    rate2, data2 = cis.wavread('./samples/voice/ilrma_f2.wav')
    rate3, data3 = cis.wavread('./samples/voice/ilrma_f3.wav')
    rate4, data4 = cis.wavread('./samples/voice/fmix_1.wav')
    sepvoice = np.array([data1, data2, data3], dtype=np.float32)
    rawvoice = np.array(data4, dtype=np.float32)

    recog = FacialRecog("samples/test.mp4", rawvoice, sepvoice, rate1, "samples/output")
    recog.set_audio()
