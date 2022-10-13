import numpy as np
from   scipy.signal import medfilt 

def Gnoisegen(signal,SNR):
    '''
    Gnoisegen 函数是叠加高斯白噪声到语音信号x中
    '''
    noise = np.random.randn(*signal.shape)
    noise = noise-np.mean(noise)
    signal_power = (1/signal.shape[0])*np.sum(np.power(signal,2))
    noise_variance = signal_power/np.power(10,(SNR/10))
    noise = (np.sqrt(noise_variance)/np.std(noise))*noise
    y = signal + noise
    return y ,noise

def frame2Time(frameNum, frame_len_s,frame_shift_s):
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
    ---------------
    >>> A = [1, 0, 0, 0, 1, 1]
    >>> find(A,'>',0)
        [0, 4, 5]
    '''
    if type(args[0])!= list:
        return print("please input a list!")
    result = []
    if len(args)!=3:
        return print('length of parameter is 3!')
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
    if express[0]==0:
        voiceIndex = find(express,0,False)
    else:
        voiceIndex = express
    record = {'begin':1,'end':1,'duration':1}
    
    soundSegment = []
    soundSegment.append(record)
    
    k = 1
    soundSegment['begin'] = voiceIndex[1]
    for i in range(voiceIndex-2):
        if voiceIndex(i+1) - voiceIndex(i)>1:
            soundSegment.append(record)
            soundSegment[k]['end'] = voiceIndex[i]
            soundSegment[k+1]['begin'] = voiceIndex[i+1]
            k = k+1
    
    soundSegment[k]['end'] = voiceIndex[len(voiceIndex)-1] # 取最后一个值
    for i in range(k):
        soundSegment['duration'] = soundSegment['end']-soundSegment['begin']
    
    return soundSegment

def VAD(dst1, T1 ,T2):
    '''
    VAD
    ----------
    语音端点检测
    
    Parameters
    ---------
    #### input:
    `dst1`:     短时能量
    `T1`:       阈值1
    `T2`:       阈值2
    #### return
    
    
    算法策略
    -------------
    
    '''
    
    
    fn = len(dst1)
    maxsilence = 8
    minlen     = 5
    status     = 0
    count   = np.zeros((1,fn))
    count = count.reshape(count.size)
    silence = np.zeros((1,fn))
    silence = silence.reshape(silence.size)
    x1 = np.zeros((1,fn))
    x1 = x1.reshape(x1.size)
    x2 = np.zeros((1,fn))
    x2 = x2.reshape(x2.size)
    # 开始端点检测
    xn = 0
    for n in range(1,fn):
        if status==1 or status ==0:
            if dst1[n] < T2:
                x1[xn] = np.max([n - count[xn]-1,1])
                status = 2
                silence[xn] = 0
                count[xn] = count[xn]+1
            elif dst1[n] <T1:
                status = 1
                count[xn] = count[xn]+1
            else:
                status = 0
                count[xn] = 0
                x1[xn] = 0
                x2[xn] = 0
        
        if status == 2:
            if dst1[n] < T1:
                count[xn] = count[xn]+1
                silence[xn] = 0
            else:
                silence[xn] = silence[xn] + 1
                if silence[xn] < maxsilence:
                    count[xn] = count[xn] + 1
                elif count[xn] < minlen:
                    status = 0
                    silence[xn] = 0
                    count[xn] = 0
                else:
                    status = 3
                    x2[xn] = x1[xn] + count[xn]
        if status == 3:
            status = 0
            xn = xn +1
            count[xn] = 0
            silence[xn] = 0
            x1[xn] = 0
            x2[xn] = 0
    el = len(x1)
    if x1[el-1] == 0:
        el = el - 1
    if el==0:
        return
    if x2[el]==0:
        print('Error: Not find endding point!\n')
        x2[el] = fn
    SF = np.zeros((1,fn))
    SF = SF.reshape(SF.size)
    NF = np.ones((1,fn))
    NF = NF.reshape(NF.size)
    
    for i in range(el):
        SF[int(x1[i]):int(x2[i])] = 1
        NF[int(x1[i]):int(x2[i])] = 0
    speechIndex = find(SF,'=',1)
    print(speechIndex)
    voiceseg = findSegment(speechIndex)
    vs1= len(voiceseg)
    return voiceseg,vs1,SF,NF

def houzhuichuli():
    return