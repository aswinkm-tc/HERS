import pyaudio
import numpy
from base import mfcc
from base import logfbank

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 1200#44100

p = pyaudio.PyAudio()
stream = p.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,
                frames_per_buffer = CHUNK)

print "* recording"
try:
    while True:
    
    	data = stream.read(CHUNK)
    	sig=numpy.fromstring(data,numpy.int16)
	mfcc_feat = mfcc(sig)
	fbank_feat = logfbank(sig)
	print fbank_feat[1:3,:]
except KeyboardInterrupt:
    pass
   
print "* done recording"
stream.close()
p.terminate()


