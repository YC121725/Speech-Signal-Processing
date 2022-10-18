import numpy as np
from   scipy.signal import medfilt 
import pandas as pd
def Gnoisegen(signal,SNR):
    '''
    叠加高斯白噪声到语音信号x中
    
    return
    ----------
    `y`
    `noise`
    '''
    noise = np.random.randn(*signal.shape)
    noise = noise-np.mean(noise)
    signal_power = (1/signal.shape[0])*np.sum(np.power(signal,2))
    noise_variance = signal_power/np.power(10,(SNR/10))
    noise = (np.sqrt(noise_variance)/np.std(noise))*noise
    y = signal + noise
    return y ,noise

def frame2Time(frameNum, frame_len_s = 0.025,frame_shift_s = 0.01):
    '''
    帧数转换为时间坐标
    '''
    frame_Time = np.linspace(0,frameNum-1,frameNum)*frame_shift_s+frame_len_s/2
    return frame_Time

def multimidfilter(x, m):
    '''
    平滑滤波
    '''
    a = x 
    for _ in range(m):
        b = medfilt(a,5)
        a = b
    y = b
    return y 

def find(*args):
    '''
    找列表中的满足表达式的元素下标
    >>> A = [1, 0, 0, 0, 1, 1]
    >>> find(A,'>',0)
        [0, 4, 5]
    '''
    if type(args[0])!= list:        # 如果不是列表就强制类型转换
        args[0] = list(args[0])
        print(type(args[0])," is already converted to list!")
    if len(args)!=3:
        return print('length of parameter is 3!')
    
    '''-----------------------------'''
    result = []
    if args[1]=='>':
        for index,i in enumerate( args[0]):
            if i>args[2]:
                result.append(index)
        return result
    if args[1]=='<':
        for index,i in enumerate( args[0]):
            if i<args[2]:
                result.append(index)
        return result
    if args[1]=='>=':
        for index,i in enumerate( args[0]):
            if i>=args[2]:
                result.append(index)
        return result
    if args[1]=='<=':
        for index,i in enumerate( args[0]):
            if i<=args[2]:
                result.append(index)
        return result
    if args[1]=='=' or args[1]=='==' :
        for index,i in enumerate( args[0]):
            if i==args[2]:
                result.append(index)
        return result
    if args[1]=='!=':
        for index,i in enumerate( args[0]):
            if i!=args[2]:
                result.append(index)
        return result
    
def findSegment(express):
    '''
    Example
    -------------------
    找音频的起点，终点，持续时间
    >>> express = # 0...1... 序列，0表示无音频，1表示有音频
    >>> findSegment(express)
    >>> return soundSegment
    [{'begin': 91, 'end': 120, 'duration': 29}, 
    {'begin': 147, 'end': 182, 'duration': 35}, 
    {'begin': 184, 'end': 282, 'duration': 98}, 
    {'begin': 336, 'end': 395, 'duration': 59}, 
    {'begin': 491, 'end': 550, 'duration': 59}, 
    {'begin': 598, 'end': 702, 'duration': 104}, 
    {'begin': 705, 'end': 796, 'duration': 91}, 
    {'begin': 812, 'end': 902, 'duration': 90}, 
    {'begin': 910, 'end': 952, 'duration': 42}]

    '''
    if express[0]==0:
        voiceIndex = find(express,'>',0)
    else:
        voiceIndex = express
    soundSegment = []
    soundSegment.append({'begin':1,'end':1,'duration':1})
    k = 0
    soundSegment[k]['begin'] = voiceIndex[0]
    for i in range(len(voiceIndex)-1):
        if voiceIndex[i+1] - voiceIndex[i]>1:
            soundSegment.append({'begin':1,'end':1,'duration':1})
            # print("更新前：",soundSegment)
            soundSegment[k]['end'] = voiceIndex[i]
            soundSegment[k+1]['begin'] = voiceIndex[i+1]
            # print("更新后：",soundSegment)
            k = k+1

    soundSegment[k]['end'] = voiceIndex[len(voiceIndex)-1] # 取最后一个值

    for i in range(k+1):
        soundSegment[i]['duration'] = soundSegment[i]['end']-soundSegment[i]['begin']
    
    return soundSegment

def VAD_H(dst1, T1 ,T2):
    '''
    VAD
    ----------
    语音端点检测
    
    Parameters
    ---------
    #### input:
    `dst1`:     短时谱熵
    `T1`:       阈值1
    `T2`:       阈值2
    #### return
    
    
    算法策略
    -------------
    (1)在
    
    '''
    fn = len(dst1)                           # 帧数

    maxsilence = 8
    minlen     = 5
    status     = 0
    
    count   = [0]
    silence = [0]
    x1 = [0]
    x2 = [0]
    
    # 开始端点检测
    xn = 0
    for n in range(fn):
        if status==1 or status ==0:
            if dst1[n] < T2:
                # print('State=1,进入语音段')
                x1[xn] = max([n - count[xn]-1,1])
                status = 2
                silence[xn] = 0
                count[xn] = count[xn]+1
            elif dst1[n] < T1:
                # print('State=1,可能处于语音段')
                status = 1
                count[xn] = count[xn]+1
            else:
                # print('State=1,静音状态')
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        
        if status == 2:
            if dst1[n] < T1:
                # print('State=2,保持在语音段')
                count[xn] = count[xn]+1
                silence[xn] = 0
            else:
                # print('State=2,语音将结束!')
                silence[xn] = silence[xn] + 1
                if silence[xn] < maxsilence:
                    count[xn] = count[xn] + 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    # print('进入State3 语音结束')
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            # print('State=3,语音结束！')
            status = 0
            xn = xn +1
            count.append(0)
            silence.append(0)
            x1.append(0)
            x2.append(0)
            
    while x1.count(0):
        x1.remove(0)
    el = len(x1)
    
    x2 = x2[0:el]
    
    if x1[el-1] == 0:
        el = el - 1
    if el==0:
        return
    if x2[el-1]==0:
        x2[el-1] = fn
    # print('x1 = ',x1)
    # print('x2 = ',x2)   
    
    SF = np.zeros((1,fn))
    SF = SF.reshape(SF.size)
    NF = np.ones((1,fn))
    NF = NF.reshape(NF.size)

    for i in range(el):
        zero = np.zeros((1,int(x2[i])-int(x1[i])))
        one  = np.ones((1,int(x2[i])-int(x1[i])))
        zero = zero.reshape(zero.size)
        one  = one.reshape(one.size)
        SF[int(x1[i]):int(x2[i])] = one
        NF[int(x1[i]):int(x2[i])] = zero
    SF = list(SF)
    
    speechIndex = find(SF,'=',1)
    voiceseg = findSegment(speechIndex)
    vs1= len(voiceseg)
    return voiceseg,vs1,SF,NF

def VAD_E(dst1, T1 ,T2):
    '''
    VAD_E
    ----------
    对短时能量进行双门限语音端点检测
    
    Parameters
    ---------
    #### input:
    `dst1`:     短时能量
    `T1`:       阈值1
    `T2`:       阈值2
    #### return
    
    '''
    fn = len(dst1)                           # 帧数

    maxsilence = 6
    minlen     = 5
    status     = 0
    
    count   = [0]
    silence = [0]
    x1 = [0]
    x2 = [0]
    
    # 开始端点检测
    xn = 0
    for n in range(fn):
        if status==1 or status ==0:
            if dst1[n] > T2:
                # print('State=1,进入语音段')
                x1[xn] = max([n - count[xn]-1,1])
                status = 2
                silence[xn] = 0
                count[xn] = count[xn]+1
            elif dst1[n] > T1:
                # print('State=1,可能处于语音段')
                status = 1
                count[xn] = count[xn]+1
            else:
                # print('State=1,静音状态')
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        
        if status == 2:
            if dst1[n] > T1:
                # print('State=2,保持在语音段')
                count[xn] = count[xn]+1
                silence[xn] = 0
            else:
                # print('State=2,语音将结束!')
                silence[xn] = silence[xn] + 1
                if silence[xn] < maxsilence:
                    count[xn] = count[xn] + 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    # print('进入State3 语音结束')
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            # print('State=3,语音结束！')
            status = 0
            xn = xn +1
            count.append(0)
            silence.append(0)
            x1.append(0)
            x2.append(0)
            
    while x1.count(0):
        x1.remove(0)
    el = len(x1)
    
    x2 = x2[0:el]
    
    if x1[el-1] == 0:
        el = el - 1
    if el==0:
        return
    if x2[el-1]==0:
        x2[el-1] = fn
    # print('x1 = ',x1)
    # print('x2 = ',x2)   
    
    SF = np.zeros((1,fn))
    SF = SF.reshape(SF.size)
    NF = np.ones((1,fn))
    NF = NF.reshape(NF.size)

    for i in range(el):
        zero = np.zeros((1,int(x2[i])-int(x1[i])))
        one  = np.ones((1,int(x2[i])-int(x1[i])))
        zero = zero.reshape(zero.size)
        one  = one.reshape(one.size)
        SF[int(x1[i]):int(x2[i])] = one
        NF[int(x1[i]):int(x2[i])] = zero
    SF = list(SF)
    
    speechIndex = find(SF,'=',1)
    voiceseg = findSegment(speechIndex)
    vs1= len(voiceseg)
    return voiceseg,vs1,SF,NF


def spliting(voiceseg):
    '''
    拼接函数
    -----------------------------
    将duration低于 20ms / 20帧 的一段去除默认为噪声干扰去除
    将两段音频间隔低于 10ms / 10帧 的两端拼接在一块

    '''    
    label = pd.DataFrame(voiceseg)
    # print('调整前：',label)       
    
    for i in range(len(voiceseg)):             
        if voiceseg[i]['duration'] < 20:   # 去掉间隔低于20ms 20/帧 
            voiceseg[i]['begin'] = 0
            voiceseg[i]['duration'] = 0
            voiceseg[i]['end'] = 0

    label = pd.DataFrame(voiceseg)
    new_label = label[(label.T!=0).any()]            # 去掉行全0的值
    new_label = new_label.reset_index(drop=True)
    voiceseg = new_label.to_dict(orient='records')  # 转换为records形式
    
    for i in range(1,len(voiceseg)):
        if voiceseg[i]['begin']-voiceseg[i-1]['end'] <11 and voiceseg[i-1]['end']!=0:
            voiceseg[i]['begin'] = voiceseg[i-1]['begin']
            voiceseg[i]['duration'] = voiceseg[i]['end']-voiceseg[i]['begin']
            voiceseg[i-1]['begin'] = 0
            voiceseg[i-1]['duration'] = 0
            voiceseg[i-1]['end'] = 0


    label = pd.DataFrame(voiceseg)
    new_label = label[(label.T!=0).any()]            # 去掉行全0的值
    new_label = new_label.reset_index(drop=True)
    # print("调整后：",new_label)
    new_label = new_label.to_dict(orient='records')  # 转换为records形式
    '''
    即:输入的原始模式
     [{'begin': 91, 'end': 120, 'duration': 29}, 
     {'begin': 147, 'end': 183, 'duration': 36}, 
     {'begin': 335, 'end': 394, 'duration': 59}, 
     {'begin': 490, 'end': 548, 'duration': 58}, 
     {'begin': 599, 'end': 777, 'duration': 178}, 
     {'begin': 813, 'end': 903, 'duration': 90}]
    '''
    return new_label