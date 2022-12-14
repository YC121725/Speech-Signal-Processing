import numpy as np
import matplotlib.pyplot as plt
"""
提供绘图函数
"""

'''-----------绘图----------'''
def plot_time(sig, fs,title):
    '''
    绘制时间图
    -----------
    '''
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
    
def plot_freq(sig, sample_rate,title='频域图', nfft=512):
    '''
    绘制频率图
    ------------
    '''
    xf = np.fft.rfft(sig, nfft)
    print('Number of fft:',(len(xf)-1)*2)      # 257个点
    xfp = -20 * np.log10(np.abs(xf))           # np.clip(np.abs(xf), 1e-20, 1e100)
    freqs = np.linspace(0, int(sample_rate/2), int(nfft/2) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, xfp)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.title(title)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()
    plt.show()

def plot_spectrogram(spec,ylabel,title):
    '''
    绘制语谱图
    ------------
    '''
    fig = plt.figure(figsize=(10, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.xlabel('Num of Frame')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def axvline(xdata,ylim):
    '''
    垂直绘图，返回横纵坐标
    
    Parameters
    --------------------
    xdata:
            [91,147,335,490,599,813]
    ylim:
            [0,10]
    
    return
    --------------------
    >>> axvline
    >>> return 
    x = [array([91., 91., 91., 91., 91., 91., 91., 91., 91., 91.]), 
    array([147., 147., 147., 147., 147., 147., 147., 147., 147., 147.]), 
    array([336., 336., 336., 336., 336., 336., 336., 336., 336., 336.]), 
    array([493., 493., 493., 493., 493., 493., 493., 493., 493., 493.]), 
    array([599., 599., 599., 599., 599., 599., 599., 599., 599., 599.]), 
    array([812., 812., 812., 812., 812., 812., 812., 812., 812., 812.])],
    y = array([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
       0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ])

    
    '''
    x = []
    y = np.linspace(ylim[0],ylim[1],10*ylim[1])
    for i in range(len(xdata)):
        x.append([])
        x[i] = np.linspace(xdata[i],xdata[i],10*ylim[1])
    
    return x,y