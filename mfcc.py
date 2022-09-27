import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from librosa import feature as lf

'''-----------预处理-------------'''
def readAudio(wav_file,iscut) :
    '''    
    预处理--读取音频
    -----------------
    wav_file:   音频名称
    iscut：     如果需要截取音频输入时长，如果不需要截取音频，输入False

    '''
    wav_file = './PHONE_001.wav'
    fs, sig = wavfile.read(wav_file)
    #fs = 8000Hz
    if iscut:
        sig = sig[0: int(iscut * fs)]    #截取音频
    else:
        sig = sig
    return fs,sig

'''-----------特征提取-------------'''
def enframe(sig=np.array([]),frame_len_s=0.025, frame_shift_s=0.01, fs=8000):
    '''
    分帧，主要是计算对应下标
    ----------------
    sig:            信号
    frame_len_s:    帧长，s 一般25ms
    frame_shift_s:  帧移，s 一般10ms
    fs:             采样率，hz

    return: 二维list，一个元素为一帧信号
    ----------------
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
    hamming
    hanning
    blackman
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



def stft(frame_sig, nfft=512 ,fs=8000,isshow_fig=False):
    """
    :param frame_sig: 分帧后的信号
    :param nfft: fft点数
    :return: 返回分帧信号的功率谱
    np.fft.fft vs np.fft.rfft
    fft 返回 nfft
    rfft 返回 nfft // 2 + 1，即rfft仅返回有效部分
    """
    frame_spec = np.fft.rfft(frame_sig, nfft)
    # 幅度谱
    fhz = np.linspace(0,frame_spec.shape[1],frame_spec.shape[1])
    fhz  = fhz*fs/nfft
    
    frame_mag = np.abs(frame_spec)
    # 功率谱
    frame_pow = (frame_mag ** 2) * 1.0 / nfft
    if isshow_fig:
        plt.figure(figsize=(10, 5))
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        plt.plot(fhz,frame_pow[1])
        plt.grid(True)
        plt.xlabel('F/Hz')
        plt.ylabel('功率谱')
        plt.title('短时傅里叶变换')
        plt.show()
    return frame_pow


def mel_filter(frame_pow, fs, n_filter, nfft, mfcc_Dimen = 12,isshow_fig = False):
    '''
    Parameter
    ------------
    mel 滤波器系数计算
    frame_pow: 分帧信号功率谱
    fs: 采样率 hz
    n_filter: 滤波器个数
    nfft: fft点数
    
    return: 
    ------------
    Filter Bank 特征和Mel特征
    '''
    mel_min = 0     # 最低mel值
    mel_max = 2595 * np.log10(1 + fs / 2.0 / 700)               # 最高mel值，最大信号频率为 fs/2
    mel_points = np.linspace(mel_min, mel_max, n_filter + 2)    # n_filter个mel值均匀分布与最低与最高mel值之间
    fhz = 700 * (10 ** (mel_points / 2595.0) - 1)               # mel值对应回频率点，频率间隔指数化
    k = np.floor(fhz * (nfft + 1) / fs)                         # 对应到fft的点数比例上
    freq = np.linspace(0,int(nfft/2),int(nfft/2+1),)*fs/nfft
    # 求mel滤波器系数
    fbank = np.zeros((n_filter, int(nfft / 2 + 1)))
    for m in range(1, 1 + n_filter):
        f_left = int(k[m - 1])     # 左边界点
        f_center = int(k[m])       # 中心点
        f_right = int(k[m + 1])    # 右边界点
        for j in range(nfft):
            if  f_left<=j and j<= f_center:
                fbank[m-1,j] = (j - f_left) / (f_center - f_left)
            elif f_center<=j and j<=f_right:
                fbank[m-1,j] = (f_right - j) / (f_right - f_center)

    if isshow_fig:
        plt.figure(figsize=(10, 5))
        for i in range(n_filter):
            plt.plot(freq,fbank[i,:])
            plt.rcParams['font.sans-serif']=['SimHei']
            plt.rcParams['axes.unicode_minus']=False
            plt.xlabel('f/Hz')
            plt.ylabel('幅值')
            plt.title('Mel滤波器组')
        plt.show()   
        
    # mel 滤波
    filter_banks = np.dot(frame_pow, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    # 取对数
    filter_banks = 20 * np.log10(filter_banks)  # dB
    # 求取MFCC特征
    mfcc = dct(filter_banks, type=2,axis=1, norm='ortho')[:, 1:(mfcc_Dimen+1)]
    return filter_banks.T,mfcc.T


def Dynamic_Feature(mfcc):
    '''
    动态特征提取
    -----------
    将MFCC进行一阶、二阶差分运算，构成动态特征
    
    Parameter
    -----------
    mfcc通常为[12,feature_num]的行向量
    '''
    delta_mfcc = lf.delta(mfcc,axis=1)
    delta2_mfcc = lf.delta(mfcc,order=2,axis=1)
    mfccs_feature = np.concatenate((mfcc,delta_mfcc,delta2_mfcc))
    return mfccs_feature


def CMVN(feature):
    '''
    倒谱均值方差归一化
    ''' 
    feat = (feature - np.mean(feature,axis=1)[:,np.newaxis])/(np.std(feature,axis=1)+np.finfo(float).eps)[:,np.newaxis]
    return feat

'''-----------绘图----------'''
def plot_time(sig, fs,title):
    time = np.arange(0, len(sig)) * (1.0 / fs)
    plt.figure(figsize=(10, 5))
    plt.plot(time, sig)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.title(title)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
    
def plot_freq(sig, sample_rate, nfft=512):
    xf = np.fft.rfft(sig, nfft)
    print('Number of fft:',(len(xf)-1)*2)     # 257个点
    xfp = -20 * np.log10(np.abs(xf))           # np.clip(np.abs(xf), 1e-20, 1e100)
    freqs = np.linspace(0, int(sample_rate/2), int(nfft/2) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, xfp)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.title('频域图')
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()
    plt.show()

def plot_spectrogram(spec,ylabel,title):
    fig = plt.figure(figsize=(10, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.xlabel('Num of Frame')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()