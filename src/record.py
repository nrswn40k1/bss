import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os
import sys

class VideoRecorder():  

    # Video class based on openCV 
    def __init__(self, device_index, save_directory, video_filename):

        self.open = True
        self.device_index = device_index
        self.fps = 30              # fps should be the minimum constant rate at which the camera can
        self.fourcc = "MJPG"       # capture images (with no decrease in speed over time; testing is required)
        self.frameSize = (640,480) # video formats and sizes also depend and vary according to the camera used
        self.video_filename = video_filename #like "temp_video.avi"
        self.save_directory = save_directory #like ./samples/group1/
        file_manager(save_directory, video_filename)
        self.video_cap = cv2.VideoCapture(self.device_index)
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.save_directory + self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.frame_counts = 1
        self.start_time = time.time()


    # Video starts being recorded 
    def record(self):

#       counter = 1
        timer_start = time.time()
        timer_current = 0


        while(self.open==True):
            ret, video_frame = self.video_cap.read()
            if (ret==True):

                    self.video_out.write(video_frame)
#                   print str(counter) + " " + str(self.frame_counts) + " frames written " + str(timer_current)
                    self.frame_counts += 1
#                   counter += 1
#                   timer_current = time.time() - timer_start
                    #time.sleep(0.033)
#                   gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
#                   cv2.imshow('video_frame', gray)
#                   cv2.waitKey(1)
            else:
                break

                # 0.16 delay -> 6 fps
                # 0.033 delay -> 30 fps


    # Finishes the video recording therefore the thread too
    def stop(self):

        if self.open==True:

            self.open=False
            self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()

        else: 
            pass


    # Launches the video recording function using a thread          
    def start(self):
        video_thread = threading.Thread(target=self.record)
        video_thread.start()





class AudioRecorder():


    # Audio class based on pyAudio and Wave
    def __init__(self, num_channels, save_directory, audio_filename):

        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = num_channels
        self.format = pyaudio.paInt16
        self.save_directory = save_directory
        self.audio_filename = audio_filename
        file_manager(save_directory, audio_filename)
        self.audio = pyaudio.PyAudio()
        self.input_device_index = self.check_device()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      input_device_index=self.input_device_index,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []

    def check_device(self):
        for x in range(0, self.audio.get_device_count()): 
            dev = self.audio.get_device_info_by_index(x)
            if dev['name'] == 'Focusrite USB ASIO' and dev['maxInputChannels'] == 18:
                return x

    # Audio starts being recorded
    def record(self):

        self.stream.start_stream()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            if self.open==False:
                break


    # Finishes the audio recording therefore the thread too    
    def stop(self):

        if self.open==True:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            waveFile = wave.open(self.save_directory + self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

        pass

    # Launches the audio recording function using a thread
    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()





def start_AVrecording(v_device_index, v_save_directory, video_filename, num_channels, a_save_directory, audio_filename):

    global video_thread
    global audio_thread

    video_thread = VideoRecorder(v_device_index, v_save_directory, video_filename)
    audio_thread = AudioRecorder(num_channels, a_save_directory, audio_filename)

    audio_thread.start()
    video_thread.start()



def start_video_recording(device_index, save_directory, video_filename):

    global video_thread

    video_thread = VideoRecorder(device_index, save_directory, video_filename)
    video_thread.start()



def start_audio_recording(num_channels, a_save_directory, audio_filename):

    global audio_thread

    audio_thread = AudioRecorder(num_channels, a_save_directory, audio_filename)
    audio_thread.start()




def stop_AVrecording():

    audio_thread.stop() 
    frame_counts = video_thread.frame_counts
    elapsed_time = time.time() - video_thread.start_time
    recorded_fps = frame_counts / elapsed_time
    print("total frames " + str(frame_counts))
    print("elapsed time " + str(elapsed_time))
    print("recorded fps " + str(recorded_fps))
    video_thread.stop() 

    # Makes sure the threads have finished
    while threading.active_count() > 1:
        time.sleep(1)


#    Merging audio and video signal
    '''
    if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected

        print("Re-encoding")
        cmd = "ffmpeg -r " + str(recorded_fps) + " -i temp_video.avi -pix_fmt yuv420p -r 6 temp_video2.avi"
        subprocess.call(cmd, shell=True)

        print("Muxing")
        cmd = "ffmpeg -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video2.avi -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)

    else:

        print("Normal recording\nMuxing")
        cmd = "ffmpeg -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video.avi -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)

        print("..")
    '''


def file_manager(directoryname, filename):
    if directoryname[-1] != '/':
        print('You are supposed to add / at the end of the save directory name.')
        sys.exit()
    else:
        path = directoryname + filename
    
    if not os.path.exists(directoryname):
        os.makedirs(directoryname)

    if os.path.exists(path):
        print('file:{} exists. Something seems to be wrong.'.format(path))
        sys.exit()

if __name__ == '__main__':
    '''
    #usage for instance, only video
    start_video_recording(2, './samples/', 'temp_video.avi')
    time.sleep(10)
    video_thread.stop()
    '''

    '''
    #usage for instance, only audio
    start_audio_recording(2, './samples/', 'temp_video.avi')
    time.sleep(10)
    audio_thread.stop()
    '''

    #usage for instance, simultaneouly
    start_AVrecording(2, './samples/', 'temp_video.avi', 2, './samples/', 'temp_audio.avi')
    time.sleep(10)
    stop_AVrecording()