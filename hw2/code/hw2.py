from ssp import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

IS = 0.25            # 前导无话段长度
wlen = 0.025         # 帧长 25ms /200
frame_shift = 0.01   # 帧移 10ms /100
SNR = 10             # 信噪比


fs, sig = readAudio('/hw2/PHONE_001.wav')
sig = sig - np.mean(sig)   # 消除直流分量
x = sig/max(abs(sig))
x = x[130000:210000]
N = len(x)
time = np.linspace(0,N,N)/fs


