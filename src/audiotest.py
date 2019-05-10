import pyaudio
import wave
import numpy as np
from scipy.io import wavfile

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 5
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=13,
                frames_per_buffer=CHUNK
                )

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()


#Not really sure what b'' means in BYTE STRING but numpy needs it
#just like wave did...
framesAll = b''.join(frames)

#Use numpy to format data and reshape.
#PyAudio output from stream.read() is interlaced.
result = np.fromstring(framesAll, dtype=np.int16)
chunk_length = len(result) // CHANNELS
result = np.reshape(result, (chunk_length, CHANNELS))

#Write multi-channel .wav file with SciPy
wavfile.write(WAVE_OUTPUT_FILENAME,RATE,result)
