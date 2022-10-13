from SSP.vad import*
from SSP.preprocess import *
from SSP.mfcc import *
import numpy as np
import matplotlib.pyplot as plt


IS = 0.25                                   # 前导无话段长度
frame_len_s = 0.025                         # 帧长 25ms /200
frame_shift_s = 0.01                        # 帧移 10ms /80
SNR = 10                                    # 信噪比
eps = np.finfo(float).eps                   # 小值

fs, sig = readAudio('./PHONE_001.wav')      
sig = sig - np.mean(sig)                    # 消除直流分量
x = sig/np.max(np.abs(sig))                 # 幅值归一化
x = x[130000:210000]                        # 取出一部分点 /80000 个点

signal ,_ = Gnoisegen(x,SNR)                # 将信号添加高斯噪声

window = Window(frame_len_s,fs,'hamming')             # 窗函数
frame_sig = enframe(signal,frame_len_s,frame_shift_s) # 分帧
frame_sig_win = (frame_sig*window).T                  # 加窗


frame_num = frame_sig_win.shape[1]          # 帧数


'''-------------短时能量-------------'''
En = []
for i in range(frame_num):
    En.append(10*np.log10(sum(frame_sig_win[:,i] * frame_sig_win[:,i])))




frameTime = frame2Time(frame_num,frame_len_s,frame_shift_s)

zcr1 = np.zeros((1,frame_num))
for i in range(frame_num):
    z = frame_sig_win[:,i]
    for j in range(int(frame_len_s*fs)-1):
        if z[j]*z[j+1]<0:
            zcr1[0][i] = zcr1[0][i]+1


df = 1/frame_len_s                           # 谱分辨率
fx1 = int(np.fix(250/df)+1);                 # 250Hz，3500Hz的位置
fx2 = int(np.fix(3500/df)+1);                # 
km = int (np.floor(frame_len_s*fs/8))        # 子带个数
K = 0.5                                     # 常熟K
Eb = []
Hb = []


for i in range(frame_num):
    A = np.abs(np.fft.fft(frame_sig_win[:,i]))      # 取来一帧数据FFT后取幅值
    E = np.zeros((int(frame_len_s*fs/2+1),1))    
    E = E.reshape(E.size)
    E[fx1+1:fx2-1] = A[fx1+1:fx2-1]                 # 只取250~3500Hz之间的分量
    E = E*E                                         # 计算能量

    P1 = E/np.sum(E)                                # 幅值归一化
    # index = find(P1,'>',0.9)                        # 寻找是否有分量的概率大于0.9
    # print(index)
    # if  index:                                      # 如果有该分量，就置于0
    #     E[index] = 0                                  
    for m in range(km):
        Eb.append(np.sum(E[4*m-3:4*m]))

    Eb = np.array(Eb)
    prob = (Eb+K)/np.sum(Eb+K)

    Hb.append(-np.sum(prob*np.log(prob+eps)))
    Eb = []

NIS = np.fix((IS-frame_len_s)/frame_shift_s*fs+1)     # 求前导无话段帧数
Enm = multimidfilter(Hb,10)
Me = np.min(Enm)
print(Me)
eth = np.mean(Enm[0:int(NIS)])
Det = eth - Me
print(Det)
T1 = 0.99*Det + Me
T2 = 0.96*Det + Me
print(T1)
print(T2)

top = Det*1.1 + Me
bottom = Me - 0.1*Det

line1 = np.linspace(T1,T1,len(frameTime))
line2 = np.linspace(T2,T2,len(frameTime))
time = np.linspace(0,len(x),len(x))/fs      # 时间刻度
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.subplot(3,1,1)
plt.plot(time,x)
plt.title('时域波形图')
plt.xlabel('时间/s')
plt.ylabel('幅值')

plt.subplot(3,1,2)
plt.plot(frameTime,Enm)
# plt.plot(frameTime,line1,'k--')
# plt.plot(frameTime,line2,'k-')
plt.title('短时改进子带谱熵')
plt.xlabel('时间/s')
plt.ylabel('谱熵值')

plt.subplot(3,1,3)
plt.plot(frameTime,En)
plt.title('短时能量')
plt.xlabel('时间/s')
plt.ylabel('幅值')
plt.show()

# voiceseg,vsl,SF,NF = VAD(Enm,T1,T2)
# for i in range(vsl):
#     nx1
