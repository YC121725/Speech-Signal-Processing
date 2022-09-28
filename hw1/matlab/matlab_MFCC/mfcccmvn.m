tic;
clear; close all; clc;

%% 预加重、分帧、加窗
frameSize = 200;  % 帧长
inc = 80;  % 帧间距
[x , fs] = audioread('PHONE_001.wav');  % 读取 PHONE_001.wav 文件
x = x(1:10000);
x = x';

N = length(x);
x = double(x);
y = zeros(1,length(x));

% 预加重 a=0.97
for i = 2:N
    y(i) = x(i)-0.97*x(i-1);
end

figure(1)
subplot(211);
xx = x(1000:1200);
yy = y(1000:1200);
plot(xx,'g'); hold on;
plot(yy,'r');
legend('一帧原始信号','一帧处理信号');
axis([0 200 -0.003 0.003])   
subplot(212);
plot(abs(fft(xx,512)),'g'); hold on;
plot(abs(fft(yy,512)),'r'); hold on;
legend('原始信号频谱','处理信号频谱');
axis([0 512 0 0.05])   

S = enframe(y,frameSize,inc);  % 分帧,对x进行分帧
[a, b] = size(S);

% 汉明窗a=0.46,得到汉明窗W=(1-a)-a*cos(2*pi*n/N)
n = 1:b;
W = 0.54-0.46*cos((2*pi.*n)/b);

% 创建汉明窗矩阵C
C = zeros(a,b);
ham = hamming(b);
for i = 1:a
    C(i,:) = ham;
end

% 将汉明窗C和S相乘得SC
SC = S.*C;

figure(2)
subplot(3,1,1); plot(C(10,:),'r');
title('汉明窗图像'); grid on;  % 画出第10帧的汉明窗图像
subplot(3,1,2);plot(S(10,:),'g');
title('原始信号图像'); grid on;  % 画出第10帧的原始信号图像
subplot(3,1,3); plot(SC(10,:),'m');
title('加了汉明窗的信号图像'); grid on;  % 画出第10帧加了汉明窗的信号图像

figure(3)
plot(S(10,:),'g');hold on;
plot(SC(10,:),'m');
legend('一段原始信号','处理后信号');

F = 0;
N = 512;
for i=1:a
    %对SC作 N=512 的FFT变换
    D(i,:)=fft(SC(i,:),N);
    %以下循环实现求取能量谱密度E
    for j=1:N
        t=abs(D(i,j));
        E(i,j)=(t^2)/N;
    end
    %获取每一帧的能量总和F(i)
    F(i)=sum(E(i,:));
    i/a;
end

%将频率转换为梅尔频率
%梅尔频率转化函数图像
mel = zeros(1,fs);
for i=1:fs
    mel(i)=2595*log10(1+i/700);
end
fl = 60;  %定义频率范围，低频
fh = 3400;%定义频率范围，高频
bl = 2595*log10(1+fl/700);%得到梅尔刻度的最小值
bh = 2595*log10(1+fh/700);%得到梅尔刻度的最大值

%梅尔坐标范围
p = 15;%滤波器个数
mm = linspace(bl,bh,p+2);%规划17个不同的梅尔刻度
fm = 700*(10.^(mm/2595)-1);%将Mel频率转换为频率
figure(4)
plot(mel,'linewidth',1.5);grid on;hold on;
plot(fm,mm,'o','color','r');
title('梅尔频率转换');xlabel('频率');ylabel('梅尔频率');

W2 = N/2+1;%fs/2内对应的FFT点数,257个频率分量
k = ((N+1)*fm)/fs%计算17个不同的k值
hm = zeros(15,N);%创建hm矩阵
df = fs/N;
freq = (0:N-1)*df;%采样频率值

%绘制梅尔滤波器
for i = 2:16
    %取整，这里取得是16个k中的第2-15个，舍弃1和17
    n0 = floor(k(i-1));
    n1 = floor(k(i));
    n2 = floor(k(i+1));
    %k(i)分别代表的是每个梅尔值在新的范围内的映射，画三角形
    for j = 1:N
       if n0<=j & j<=n1
           hm(i-1,j)=(j-n0)/(n1-n0);
       elseif n1<=j & j<=n2
           hm(i-1,j)=(n2-j)/(n2-n1);
       end
    end 
end

%绘图,且每条颜色显示不一样
c = colormap(lines(15));%定义15条不同颜色的线条
figure(5)
for i = 1:15
    plot(freq,hm(i,:),'--','color',c(i,:),'linewidth',2);%开始循环绘制每个梅尔滤波器
    hold on
end
 grid on;%显示方格
 axis([0 4000 0 1]);%设置显示范围
 
 %得到能量特征参数的和
 H = E*hm';
 
 %对H作自然对数运算
 %因为人耳听到的声音与信号本身的大小是幂次方关系，所以要求个对数
 for i = 1:a
     for j = 1:15
         H(i,j) = log(H(i,j));
     end
 end

%作离散余弦变换DCT   计算MFCC参数
mfcc = zeros(a,15);
for i = 1:a
    for j = 1:15
        %先求取每一帧的能量总和
        sum1 = 0;
        %作离散余弦变换
        for p = 1:15
            sum1 = sum1+H(i,p)*cos((pi*j)*(2*p-1)/(2*15));
        end
        mfcc(i,j) = ((2/15)^0.5)*sum1;     %MFCC参数
        %完成离散余弦变换
    end    
end
 
 
% 差分计算
% 作升倒谱运算，归一化倒谱提高窗口
% 因为大部分的信号数据一般集中在变换后的低频区，所以对每一帧只取前13个数据就好了
J = mfcc(:,(1:13));
K = zeros(1,13);
for i=1:13
    K(i)=1+(22/2)*sin(pi*i/22);
end
K = K/max(K);
 
% 得到二维数组m,这是mfcc的原始数据;
L = zeros(a,13);
for i = 1:a
    for j = 1:13
        L(i,j) = J(i,j)*K(j);
    end
end

feat = L;
% 接下来求取mfcc参数的三阶差分参数 48584x13
% 接下来求取第二组（一阶差分系数）48584x13，这也是mfcc参数的第二组参数
% dtfeat = 0;
dtfeat = zeros(size(L));%默认初始化
for i=3:a-2
    dtfeat(i,:) = -2*feat(i-2,:)-feat(i-1,:)+feat(i+1,:)+2*feat(i+2,:); 
end
dtfeat = dtfeat/10;

% 求取二阶差分系数,mfcc参数的第三组参数
% 二阶差分系数就是对前面产生的一阶差分系数dtfeat再次进行操作。
% dttfeat = 0;
dttfeat = zeros(size(dtfeat));%默认初始化
for i=3:a-2
    dttfeat(i,:) = -2*dtfeat(i-2,:)-dtfeat(i-1,:)+dtfeat(i+1,:)+2*dtfeat(i+2,:); 
end
dttfeat = dttfeat/10;

% 求取三阶差分系数,mfcc参数的第四组参数
% 三阶差分系数就是对前面产生的一阶差分系数dtfeat再次进行操作。
% dtttfeat = 0;
dtttfeat = zeros(size(dtfeat));%默认初始化
for i=3:a-2
    dtttfeat(i,:) = -2*dttfeat(i-2,:)-dttfeat(i-1,:)+dttfeat(i+1,:)+2*dttfeat(i+2,:); 
end
dtttfeat = dtttfeat/10;

%将得到的mfcc的三个参数feat、dtfeat、dttfeat拼接到一起
%得到最后的mfcc系数48584x52
% mfcc_final = 0;
mfcc_final = [feat,dtfeat,dttfeat,dtttfeat];%拼接完成
disp(size(feat))
disp(size(dtfeat))
disp(size(mfcc_final))
% % 除去首尾两帧，因为这两帧的差分参数为0
% dtmm = dtm(3:size(m,1)-2,:)

% cepstral mean and variance normalization (CMVN)
% cmvn = zscore(mfcc_final);
% 
% mfcc1 = mfcc';
% t1 = linspace(0,size(mfcc1,2)*80/8000,485);
% f1 = linspace(1,size(mfcc1,1),15);
% figure(6);
% imagesc(t1, f1, 20*log10((abs(mfcc1))));xlabel('s'); ylabel('db');
% colorbar;
% title('MFCC(1-15)维');
% 
% mfcc2 = mfcc';
% mfcc2 = mfcc2(1:14,:);
% t2 = linspace(0,size(mfcc2,2)*80/8000,485);
% f2 = linspace(1,size(mfcc2,1)-1,14);
% figure(7);
% imagesc(t2, f2, 20*log10((abs(mfcc2))));xlabel('s'); ylabel('db');
% colorbar;
% title('MFCC(1-14)维');
% 
% mfcc_final1 = mfcc_final';
% mfcc_final1 = mfcc_final1(14:52,:);
% t3 = linspace(0,size(mfcc_final,1)*80/8000,485);
% f3 = linspace(14,size(mfcc_final,2),39);
% figure(8);
% imagesc(t3, f3, 20*log10((abs(mfcc_final1))));xlabel('s'); ylabel('db');
% colorbar;
% title('MFCC差分(14-52)维');
% 
% cmvn1 = cmvn';
% cmvn1 = cmvn1(14:52,:);
% t4 = linspace(0,size(cmvn,1)*80/8000,485);
% f4 = linspace(14,size(cmvn,2),39);
% figure(9);
% imagesc(t4, f4, 20*log10((abs(cmvn1))));xlabel('s'); ylabel('db');
% colorbar;
% title('MFCC差分-CMVN(14-52)维');
% 
% toc;