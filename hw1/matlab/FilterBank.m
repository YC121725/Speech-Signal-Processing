function [ triangle ] = FilterBank(fs,p,N)
% [data,fs] = audioread('PHONE_001.wav');

clc;
clear;
close all;
fs = 8000;  %信号采样频率
p = 15;   %梅尔滤波器个数
N = 1024;
fl=0;fh=fs/2;%定义频率范围，低频和高频
bl=2595*log10(1+fl/700);%得到梅尔刻度的最小值
bh=2595*log10(1+fh/700);%得到梅尔刻度的最大值
B=bh-bl;%梅尔刻度长度
mm=linspace(0,B,p+2);%规划18个不同的梅尔刻度。但只有16个梅尔滤波器
fm=700*(10.^(mm/2595)-1);%将Mel频率转换为频率

k=((N+1)*fm)/fs;%计算18个不同的k值
hm=zeros(p,N);%创建hm矩阵
df=fs/N;
freq=(0:N-1)*df;%采样频率值

%绘制梅尔滤波器
for i=2:p+1
    %取整
    n0=floor(k(i-1));  %floor为向下取整，如果取整，表示三角滤波器组的各中心点被置于FFT 数据对应的的点位频率上，所以频率曲线的三角形顶点正好在FFT 数据点上，即美尔滤波器所有顶点在同一高度。
    n1=floor(k(i));
    n2=floor(k(i+1));
%     n0=k(i-1);  %不取整代表直接按精确指定的中心频率点来计算的，这时滤波器中心频率就不一定在FFT 数据点对应位置，按FFT 点位所得曲线在低频段就不是完美的三角形，即美尔滤波器所有顶点不在同一高度。
%     n1=k(i);
%     n2=k(i+1);
    %要知道k(i)分别代表的是每个梅尔值在新的范围内的映射，其取值范围为：0-N/2
    %以下实现求取三角滤波器的频率响应。
    
    for j=1:N
       if n0<=j & j<=n1
           hm(i-1,j)=(j-n0)/(n1-n0);
       elseif n1<=j & j<=n2
           hm(i-1,j)=(n2-j)/(n2-n1);
       end
   end
   %此处求取hm结束。
end

%绘图,且每条颜色显示不一样
c=colormap(lines(p));%定义16条不同颜色的线条
% figure()
% set(gcf,'position',[180,160,1020,550]);%设置画图的大小
for i=1:p
    plot(freq,hm(i,:),'-','color',c(i,:),'linewidth',2);%开始循环绘制每个梅尔滤波器
%     plot(freq,hm(i,:),'-','linewidth',2.5);%开始循环绘制每个梅尔滤波器
    hold on;
end
grid on;%显示方格
axis([0 fh 0 inf]);%设置显示范围
xlabel('频率（Hz）');
ylabel('幅值');

% return ;
end