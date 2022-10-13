"""
SSP is Speach Signal Processing
"""
__version__ ='0.0.1'
__author__='Yangchen'
__doc__ = """
SSP 
----------
语音信号处理工具包
"""
from .preprocess import(
    readAudio,
    enframe,
    Window   
)
from .mfcc import(
    stft,
    mel_filter,
    Dynamic_Feature,
    CMVN
)
from .sspplot import(
    plot_time,
    plot_freq,
    plot_spectrogram
)
from .vad import(
    Gnoisegen,
    frame2Time
)