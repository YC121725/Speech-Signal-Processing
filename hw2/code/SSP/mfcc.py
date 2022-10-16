import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt

"""
包含MFCC各种处理函数

"""
def stft(frame_sig, nfft=512 ,fs=8000,isshow_fig = False):
    """
    短时傅里叶变换
    ---------------
    >>> stft(frame_sig,512,8000)
    >>> return frame_pow
    
    Parameters
    ---------------
    `frame_sig`: 分帧后的信号
    `nfft`: fft点数
    `fs`: 采样率
    `isshow_fig`:显示第一帧的stft图像
    
    说明
    ---------------
    >>> np.fft.fft vs np.fft.rfft
    >>> np.fft.fft
    >>>     return nfft
    >>> np.fft.rfft
    >>>     return nfft/2 + 1
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
        plt.plot(fhz,frame_pow[0])
        plt.grid(True)
        plt.xlabel('F/Hz')
        plt.ylabel('功率谱')
        plt.title('短时傅里叶变换')
        plt.show()
    return frame_pow

def mel_filter(frame_pow, fs, n_filter, nfft, mfcc_Dimen = 12,isshow_fig = False):
    '''
    mel滤波器系数计算
    ----------------
    >>> mel_filter(frame_pow,fs=8000,n_filter=15,nfft=512,mfcc_Dimen = 13)
    >>>     return filter_bank,mfcc,Mel_Filters
    
    Parameters
    ------------
    `frame_pow`:  分帧信号功率谱
    `fs`:         采样率 hz
    `n_filter`:   滤波器个数
    `nfft`:       fft点数,通常为512
    `mfcc_Dimenson`: 取多少DCT系数，通常为12-13

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
    mfcc = dct(filter_banks, type=2,axis=1, norm='ortho')[:, 1:mfcc_Dimen+1]
    return filter_banks.T,mfcc.T,fbank

def Dynamic_Feature(mfcc,cutframe = True):
    '''
    动态特征提取
    -----------
    包含升倒谱系数;三阶差分运算

    (1)提升倒谱系数可以提高在噪声环境下的识别效果
    >>> Dynamic_Feature(mfcc)  # mfcc [] 
    >>>     return mfcc_final 
    
    (2)将经过升倒谱后的MFCC进行一阶、二阶差分运算、三阶差分运算，构成动态特征
    自动切除前后两帧
    
    Parameters
    -----------
    `mfcc`：        通常为[MFCC,feature_num]的行向量    
    `cutframe`：    是否切除前后各两帧
    ''' 
    J = mfcc.T
    K = []
    for i in range(J.shape[1]):
        K.append(1+ 22/2*np.sin(np.pi*i/22))
    K = K/np.max(K)
    feat = np.zeros((J.shape[0],J.shape[1]))

    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            feat[i][j] = J[i][j]*K[j]
    
    '''--------3阶差分运算--------
        一阶差分
    '''
    dtfeat = np.zeros(feat.shape)
    for i in range(2,dtfeat.shape[0]-2):
        dtfeat[i,:] = -2*feat[i-2,:]-feat[i-1,:]+feat[i+1,:]+2*feat[i+2,:]
    dtfeat = dtfeat/10
    
    '''二阶差分'''
    dttfeat = np.zeros(feat.shape)
    for i in range(2,dttfeat.shape[0]-2):
        dttfeat[i,:] = -2*dtfeat[i-2,:]-dtfeat[i-1,:]+dtfeat[i+1,:]+2*dtfeat[i+2,:]
    dttfeat = dttfeat/10
    
    '''二阶差分'''
    dtttfeat = np.zeros(feat.shape)
    for i in range(2,dtttfeat.shape[0]-2):
        dtttfeat[i,:] = -2*dttfeat[i-2,:]-dttfeat[i-1,:]+dttfeat[i+1,:]+2*dttfeat[i+2,:]
    dtttfeat = dtttfeat/10   
    
    '''拼接'''
    mfcc_final = np.concatenate((feat.T,dtfeat.T,dttfeat.T,dtttfeat.T))
    
    '''是否去掉前后各两帧'''
    if cutframe:
        mfcc_final = mfcc_final[2:-2,:]
        return mfcc_final
    else:
        return mfcc_final

def CMVN(feature):
    '''
    倒谱均值方差归一化
    ''' 
    feat = (feature - np.mean(feature,axis=1)[:,np.newaxis])/(np.std(feature,axis=1)+np.finfo(float).eps)[:,np.newaxis]
    return feat


