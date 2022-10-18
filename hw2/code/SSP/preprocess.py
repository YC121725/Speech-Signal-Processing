import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

'''-----------预处理-------------'''
def readAudio(wav_file,iscut=False) :
    '''    
    预处理——读取音频
    -----------------
    `wav_file`:   音频名称
    `iscut`：     如果需要截取音频输入时长，如果不需要截取音频，输入False

    '''
    fs, sig = wavfile.read(wav_file)
    #fs = 8000Hz
    if iscut:
        sig = sig[0: int(iscut * fs)]    #截取音频
    else:
        sig = sig
    return fs,sig

def writeAudio(filename,sig,fs,voiceseg,frame_shift_s=0.01,frame_len_s=0.025):
    
    for i in range(len(voiceseg)):
        begin = np.round((voiceseg[i]['begin'] * frame_shift_s+frame_len_s/2)*fs)
        end = np.round((voiceseg[i]['end'] * frame_shift_s+frame_len_s/2)*fs)
        wavfile.write('./{}/PHONE_001_{}_{}.wav'.format(filename,voiceseg[i]['begin']*10,voiceseg[i]['end']*10),
                    fs,sig[int(begin):int(end)].astype(np.int16))

'''-----------特征提取-------------'''
def enframe(sig=np.array([]),frame_len_s=0.025, frame_shift_s=0.01, fs=8000):
    '''
    分帧
    ----------------
    主要是计算对应下标,并重组
    >>> enframe(sig,0.025,0.01,8000)
    >>>     return frame_sig
    
    Parameters
    ----------------
    `sig`:            信号
    `frame_len_s`:    帧长，s 一般25ms
    `frame_shift_s`:  帧移，s 一般10ms
    `fs`:             采样率，hz

    return 
    ----------------
    二维list，一个元素为一帧信号
    '''
    # np.round 四舍五入
    # np.ceil 向上取整
    sig_n = len(sig)

    frame_len_n, frame_shift_n = int(round(fs * frame_len_s)), int(round(fs * frame_shift_s))
    print('frame_len_n:',frame_len_n,'\t','frame_shift_n:',frame_shift_n)
    
    num_frame = int(np.ceil(float(sig_n - frame_len_n) / frame_shift_n))
    print('length of Sig:',sig_n,'\t','num_frame:',num_frame)
    
    pad_num = frame_shift_n * num_frame + frame_len_n - sig_n   # 待补0的个数
    pad_zero = np.zeros(int(pad_num))    # 补0
    pad_sig = np.append(sig, pad_zero)   

    index = np.arange(0,frame_len_n)                           # 一行    
    frame_index = np.tile(index,(num_frame,1))                 # frame num个行
    '''
    print(frame_index)
    [[  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]
    ...
    [  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]
    [  0   1   2 ... 197 198 199]]
    '''
    frame_inner_index = np.arange(0,num_frame)*frame_shift_n   # frame num 个值
    frame_inner_index_expand = np.expand_dims(frame_inner_index,1)
    '''
    print(frame_inner_index_expand)
    [[    0]
    [   80]
    ...
    [27760]]
    '''
    each_frame_index = frame_inner_index_expand + frame_index
    each_frame_index = each_frame_index.astype(int,copy=False)
    '''
    print(each_frame_index)
    [[    0     1     2 ...   197   198   199]
    [   80    81    82 ...   277   278   279]
    [  160   161   162 ...   357   358   359]
    ...
    [27600 27601 27602 ... 27797 27798 27799]
    [27680 27681 27682 ... 27877 27878 27879]
    [27760 27761 27762 ... 27957 27958 27959]]
    '''
    frame_sig = pad_sig[each_frame_index]
    # print(frame_sig.shape)
    return frame_sig  

def Window(frame_len_s,fs,Window_Type,isshow_fig=False):
    '''
    三种窗函数
    -------
    
    Parameters
    ---------
    `frame_len_s`:    0.025
    `fs`:             8000
    `Window_Type`:    `hamming`;`hanning`;`blackman`
    '''
    if Window_Type =='hamming':
        window = np.hamming(int(round(frame_len_s * fs)))
    if Window_Type =='hanning':
        window = np.hanning(int(round(frame_len_s * fs)))
    if Window_Type =='blackman':
        window = np.blackman(int(round(frame_len_s * fs)))   
    if isshow_fig: 
        plt.figure(figsize=(10, 5))
        plt.plot(window)
        plt.grid(True)
        plt.xlim(0, 200)
        plt.ylim(0, 1)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title(Window_Type)
        plt.show()
    return window