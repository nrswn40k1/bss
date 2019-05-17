import pyaudio
import wave
import numpy as np
from scipy.io import wavfile

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 3
RATE = 44100
RECORD_SECONDS = 10
group_number = '5'
WAVE_OUTPUT_FILENAME = "./samples/group{}/output.wav".format(group_number)

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=14,
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

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
