import numpy as np
from scipy.fftpack import dct
from mfcc import *


if __name__ =='__main__':

    '''--------读取数据--------
       截取5s的音频    
    '''
    fs, sig = readAudio('./PHONE_001.wav',5)
    plot_time(sig, fs,'原始信号')
    
    '''--------预处理--------'''
    '''(1)预加重'''
    alpha = 0.97
    sig = np.append(sig[0], sig[1:] - alpha * sig[:-1])
    plot_time(sig, fs,'预加重信号')
    plot_freq(sig, fs) 

    '''(2)分帧'''
    frame_len_s = 0.025    #25ms
    frame_shift_s = 0.01   #10ms
    frame_sig = enframe(sig,frame_len_s, frame_shift_s, fs)
    plot_time(frame_sig[1], fs,'加窗前第一帧时域图')
    
    '''(3)加窗'''
    window = Window(frame_len_s,fs,'hamming')
    frame_sig_win = window * frame_sig
    plot_time(frame_sig_win[1], fs,'加窗后第一帧时域图')


    '''--------stft--------'''
    nfft = 512
    frame_pow = stft(frame_sig, nfft ,fs)

    '''--------Mel 滤波器组--------'''
    '''Filter Bank 特征和MFCC特征提取'''

    n_filter = 16   # mel滤波器个数
    filter_banks,mfcc = mel_filter(frame_pow, fs, n_filter, nfft,mfcc_Dimen=12,isshow_fig=False)
    plot_spectrogram(filter_banks,'Filter Banks','Filter Banks特征')
    # 去均值
    filter_banks -= (np.mean(filter_banks, axis=1)[:,np.newaxis] + np.finfo(float).eps)
    plot_spectrogram(filter_banks,'Filter Banks','去均值后特征')
    plot_spectrogram(mfcc, 'MFCC系数','MFCC 特征')
    
    '''动态特征提取'''
    mfccs_feature = Dynamic_Feature(mfcc)
    plot_spectrogram(mfccs_feature, 'MFCC系数','动态MFCC特征')    
    
    '''CMVN'''
    feat = CMVN(mfccs_feature)
    print(feat.shape)
    plot_spectrogram(feat, 'MFCC系数','CMVN')
    




