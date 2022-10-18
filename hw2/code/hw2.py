from SSP.vad import*
from SSP.preprocess import *
from SSP.sspplot import *
import numpy as np
import matplotlib.pyplot as plt


''' ---------------参数设置--------------'''
IS = 0.25                                   # 前导无话段长度
frame_len_s = 0.025                         # 帧长 25ms /200
frame_shift_s = 0.01                        # 帧移 10ms /80
SNR = 10                                    # 信噪比
eps = np.finfo(float).eps                   # 小值


'''----------------预处理----------------'''
fs, sig = readAudio('./hw2/PHONE_001.wav',iscut=False)    
sig_ = sig - np.mean(sig)                    # 消除直流分量
x = sig_/np.max(np.abs(sig_))                # 幅值归一化
signal ,_ = Gnoisegen(x,SNR)                 # 将信号添加高斯噪声


'''----------------分帧 加窗----------------'''
window = Window(frame_len_s,fs,'hamming')             # 窗函数
frame_sig = enframe(signal,frame_len_s,frame_shift_s) # 分帧
frame_sig_win = (frame_sig*window).T                  # 加窗
frame_num = frame_sig_win.shape[1]                    # 帧数
km = int (np.floor(frame_len_s*fs/8))                 # 子带个数 200/8 = 25 
K = 0.5


'''-------------短时能量-------------'''
En = []
for i in range(frame_num):
    En.append(10*np.log10(sum(frame_sig_win[:,i] * frame_sig_win[:,i])))

En = multimidfilter(En,10)


'''------------计算过零率------------'''
'''
浊音的过零率:低
清音的过零率:高
'''
zcr = np.zeros((1,frame_num))
zcr = zcr.reshape(zcr.size)
for i in range(frame_num):
    z = frame_sig_win[:,i]
    for j in range(int(frame_len_s*fs)-1):
        if z[j]*z[j+1]<0:
            zcr[i] = zcr[i]+1

zcr_ = multimidfilter(zcr,10)
    

'''--------------计算短时幅度谱熵--------------'''

df = 1/frame_len_s                           # 谱分辨率 == 40 1个点代表40Hz
fx1 = int(np.fix(250/df)+1);                 # 250Hz，3500Hz的位置
fx2 = int(np.fix(3500/df)+1);                # 
km = int (np.floor(frame_len_s*fs/8))        # 子带个数 200/8 = 25 
K = 0.5                                      # 常熟K
Eb = []
Hb = []

for i in range(frame_num):
    A = np.abs(np.fft.rfft(frame_sig_win[:,i]))       # 取来一帧数据FFT后取幅值 RFFT
    E = A[fx1+1:fx2-1]**2                             # 只取250~3500Hz之间的分量 计算能量 
    for m in range(km):
        Eb.append(np.sum(E[4*m-3:4*m]))
    Eb = np.array(Eb)
    prob = (Eb+K)/np.sum(Eb+K)
    Hb.append(-np.sum(prob*np.log(prob+eps)))
    Eb = []
# Enm = multimidfilter(Hb,10)


''' ----------- 采用谱熵VAD检测 --------------'''
# NIS = np.fix((IS-frame_len_s)/frame_shift_s*fs+1)     # 求前导无话段帧数
# Me = np.min(Enm)
# eth = np.max(Enm[0:int(NIS)])
# Det = eth - Me

# T1 = 0.99*Det + Me
# T2 = 0.96*Det + Me

# time = np.linspace(0,len(x),len(x))/fs                # 时间刻度
# frameTime = frame2Time(frame_num,frame_len_s,frame_shift_s)
# line1 = np.linspace(T1,T1,len(frameTime))
# line2 = np.linspace(T2,T2,len(frameTime))

# voiceseg,vsl,SF,NF =  VAD_H(Enm,T1,T2)

# voiceseg = spliting(voiceseg)
# print('length(voiceseg):',len(voiceseg))


''' -----------采用能量 VAD检测 --------------'''
# NIS_ = np.fix((IS-frame_len_s)/frame_shift_s*fs+1)     # 求前导无话段帧数
Me = np.min(En)
T1 = 0.9*Me
T2 = T1+0.6

voiceseg,vsl,SF,NF =  VAD_E(En,T1,T2)

voiceseg = spliting(voiceseg)
print('length(voiceseg):',len(voiceseg))


''' ----------- 绘图 --------------'''

time = np.linspace(0,len(x),len(x))/fs                 # 时间刻度
frameTime = frame2Time(frame_num,frame_len_s,frame_shift_s)
line1 = np.linspace(T1,T1,len(frameTime))
line2 = np.linspace(T2,T2,len(frameTime))

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.subplot(3,1,1)
plt.plot(time,x)
plt.title('时域图')
plt.xlabel('时间/s')
plt.ylabel('幅值')

plt.subplot(3,1,2)
plt.plot(frameTime,En)
plt.plot(frameTime,line1,'k--')
plt.plot(frameTime,line2,'k-')
plt.title('短时能量')
plt.xlabel('时间/s')
plt.ylabel('谱熵值')

plt.subplot(3,1,3)
plt.plot(frameTime,Hb)
plt.title('改进短时幅度谱熵')
plt.xlabel('时间/s')
plt.ylabel('谱熵值')

plt.subplots_adjust(hspace=0.7)


'''--------------绘制直线图--------------'''

begin = np.ones((1,len(voiceseg)))
begin = list(begin.reshape(begin.size))

end = np.ones((1,len(voiceseg)))
end = list(end.reshape(end.size))

for i in range(len(voiceseg)):
    begin[i] = voiceseg[i]['begin']
    end[i] = voiceseg[i]['end']


x1 , y1 = axvline(begin,[-int(max(x)),int(max(x))])
x2 , y2 = axvline(end,[-int(max(x)),int(max(x))])

Hbx1,Hby1 = axvline(begin,[int(min(Hb)),int(max(Hb))])
Hbx2,Hby2 = axvline(end,[int(min(Hb)),int(max(Hb))])

Enx1,Eny1 = axvline(begin,[int(min(En)),int(max(En))])
Enx2,Eny2 = axvline(end,[int(min(En)),int(max(En))])

zcrx1,zcry1 = axvline(begin,[int(min(zcr)),int(max(zcr))])
zcrx2,zcry2 = axvline(end,[int(min(zcr)),int(max(zcr))])

for i in range(len(x1)):
    plt.subplot(3,1,1)
    plt.plot(x1[i] * frame_shift_s , y1 ,'r--')
    plt.plot(x2[i] * frame_shift_s , y2 ,'k--')
    
    plt.subplot(3,1,2)
    plt.plot(Enx1[i] * frame_shift_s , Eny1 ,'r--')
    plt.plot(Enx2[i] * frame_shift_s , Eny2 ,'k--')
    
    plt.subplot(3,1,3)
    plt.plot(Hbx1[i] * frame_shift_s , Hby1 ,'r--')
    plt.plot(Hbx2[i] * frame_shift_s , Hby2 ,'k--')

plt.show()

'''--------------切分音频--------------'''

writeAudio('hw2/audio',sig,fs,voiceseg,frame_shift_s=0.01,frame_len_s=0.025)

