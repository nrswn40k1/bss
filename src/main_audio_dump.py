"""
recording の際，必要なデバイスのインデックスがわからなければこれを実行する
"""

import pyaudio

def audio_dump():
    pa = pyaudio.PyAudio()
    num_api = pa.get_host_api_count()
    num_dev = pa.get_device_count()

    api_info = []
    for i in range(0,num_api):
        api_info.append(pa.get_host_api_info_by_index(i))

    for i in range(0,num_dev):
        dev_info = pa.get_device_info_by_index(i)
        print("Index: %d" % i)
        print(" -DeviceName: %s" % dev_info['name'])
        print(" -HostName: %s" % api_info[int(dev_info['hostApi'])]['name'])
        print(" -SampleRate: %dHz\n -MaxInputChannels: %d\n -MaxOutputChannels: %d" % (int(dev_info['defaultSampleRate']), int(dev_info['maxInputChannels']), int(dev_info['maxOutputChannels'])))


if __name__ == '__main__':
    audio_dump()
