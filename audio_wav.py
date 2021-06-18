# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
  
# Sampling frequency
freq = 48100
  
# Recording duration
duration = 1
  
# Start recorder with the given values 
# of duration and sample frequency

print("Ya estoy grabando, ponte verga")
recording = sd.rec(int(duration * freq), 
                   samplerate=freq, channels=2)
  
# Record audio for the given number of seconds
sd.wait()
  

  
# Convert the NumPy array to audio file
wv.write("recording1.wav", recording, freq, sampwidth=2)
