import numpy as np
from scipy.fftpack import dct
from mfcc import *
from librosa import feature as lf

if __name__ =='__main__':

    '''--------读取数据--------
       截取5s的音频    
    '''
    fs, sig = readAudio('./PHONE_001.wav',5)

    '''--------预处理--------'''
    '''(1)预加重'''
    alpha = 0.97
    sig = np.append(sig[0], sig[1:] - alpha * sig[:-1])
    # plot_time(sig, fs)
    # plot_freq(sig, fs) 

    '''(2)分帧'''
    frame_len_s = 0.025    #25ms
    frame_shift_s = 0.01   #10ms
    frame_sig = enframe(sig,frame_len_s, frame_shift_s, fs)

    '''(3)加窗'''
    window = Window(frame_len_s,fs,'hamming')
    frame_sig_win = window * frame_sig

    # plot_time(frame_sig[1], fs,'加窗前第一帧时域图')
    # plot_time(frame_sig_win[1], fs,'加窗前第一帧时域图')


    '''--------stft--------'''
    nfft = 512
    frame_pow = stft(frame_sig, nfft ,fs)

    '''--------Mel 滤波器组--------'''
    '''(1)Filter Bank 特征'''

    n_filter = 16   # mel滤波器个数
    filter_banks = mel_filter(frame_pow, fs, n_filter, nfft,isshow_fig=False)
    # plot_spectrogram(filter_banks.T,'Filter Banks','Filter Banks特征')
    # print(filter_banks.shape)
    # 去均值
    filter_banks -= (np.mean(filter_banks, axis=0) + np.finfo(float).eps)
    # plot_spectrogram(filter_banks.T,'Filter Banks','去均值后特征')

    '''--------DCT--------'''
    '''(2)MFCC 特征'''
    mfcc_Dimen = 12
    mfcc = dct(filter_banks, type=2,axis=1, norm='ortho')[:, 1:(mfcc_Dimen+1)]
    # plot_spectrogram(mfcc.T, 'MFCC系数','MFCC 特征')
    
    '''动态特征提取'''
    delta_mfcc = lf.delta(mfcc.T,axis=1)
    delta2_mfcc = lf.delta(mfcc.T,order=2,axis=1)
    mfccs_feature = np.concatenate((mfcc.T,delta_mfcc,delta2_mfcc))
    print(mfccs_feature.shape)
    plot_spectrogram(mfccs_feature, 'MFCC系数','动态MFCC特征')
    




