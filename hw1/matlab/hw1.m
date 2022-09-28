% 
% clc;
% clear;
% %读取音频
% [signal,fs] = audioread('PHONE_001.wav');
% disp(fs)
% 
% t = length(signal)/fs;              %计算持续时间
% 
% %% FFT 
% sample_points = linspace(1/fs,t,length(signal));       %计算采样点横坐标
% fhz = linspace(-fs/2,fs/2,length(signal));             %FFT横坐标转换
% 
% freq_signal=fft(signal);              %计算频谱信号
% % freq_signal=fftshift(freq_signal);  %频谱搬迁
% abs_fft = abs(freq_signal);           %取绝对值
% fft_length = length(abs(freq_signal));%长度
% disp(fft_length)
% 
% % 绘制原始信号时域波形
% figure(1)
% subplot(311);
% plot(sample_points,signal); 
% title('Time domain')
% xlabel('Time/s');
% ylabel('Amplitude');
% %绘制原始信号时域波形
% subplot(312);
% plot(fhz,abs(freq_signal)); 
% title('Frequency domain')
% xlabel('Frequency/Hz');
% ylabel('Amplitude');
% 
% subplot(313);
% plot(linspace(0,fs/2,fft_length/2),abs_fft(1:fft_length/2)); 
% title('Frequency domain')
% xlabel('Frequency/Hz');
% ylabel('Amplitude');
% 
% 
% %% 三角滤波器组
% 
% % fs = 8000;                %信号采样频率
% p = 15;                     %梅尔滤波器个数
% N = 1024;
% 
% fl=0; fh=fs/2;              %定义频率范围，低频和高频
% 
% bl = 2595*log10(1+fl/700);  %得到梅尔刻度的最小值
% bh = 2595*log10(1+fh/700);  %得到梅尔刻度的最大值
% 
% B  = bh-bl;                 %梅尔刻度长度
% mm = linspace(0,B,p+2);     %规划17个不同的梅尔刻度。但只有15个梅尔滤波器
% fm = 700*(10.^(mm/2595)-1); %将Mel频率转换为频率
% 
% k  = ((N+1)*fm)/fs;         %计算18个不同的k
% 
% hm = zeros(p,N);            %创建hm矩阵
% df = fs/N;
% freq = (0:N-1)*df;          %采样频率值
% 
% %绘制梅尔滤波器
% for i=2:p+1
%     %取整
%     n0=floor(k(i-1));  %floor为向下取整，如果取整，表示三角滤波器组的各中心点被置于FFT 数据对应的的点位频率上，所以频率曲线的三角形顶点正好在FFT 数据点上，即美尔滤波器所有顶点在同一高度。
%     n1=floor(k(i));
%     n2=floor(k(i+1));
% %     n0=k(i-1);  %不取整代表直接按精确指定的中心频率点来计算的，这时滤波器中心频率就不一定在FFT 数据点对应位置，按FFT 点位所得曲线在低频段就不是完美的三角形，即美尔滤波器所有顶点不在同一高度。
% %     n1=k(i);
% %     n2=k(i+1);
%     %要知道k(i)分别代表的是每个梅尔值在新的范围内的映射，其取值范围为：0-N/2
%     %以下实现求取三角滤波器的频率响应。
%     
%     for j=1:N
%        if n0<=j & j<=n1
%            hm(i-1,j)=(j-n0)/(n1-n0);
%        elseif n1<=j & j<=n2
%            hm(i-1,j)=(n2-j)/(n2-n1);
%        end
%    end
%    %此处求取hm结束。
% end
% 
% %绘图,且每条颜色显示不一样
% c=colormap(lines(p));%定义16条不同颜色的线条
% % figure()
% % set(gcf,'position',[180,160,1020,550]);%设置画图的大小
% figure(2)
% for i=1:p
%     plot(freq,hm(i,:),'-','color',c(i,:),'linewidth',2);%开始循环绘制每个梅尔滤波器
% %     disp(size(hm(i,:)))
% %     plot(freq,hm(i,:),'-','linewidth',2.5);%开始循环绘制每个梅尔滤波器
%      hold on;
% end
% for i =1:p
%     if i==1
%         new = hm(1,:);
%     else
%         new = hm(i,:)+new;
%     end
% end
% disp(length(new))
% grid on;%显示方格
% axis([0 fh 0 inf]);%设置显示范围
% xlabel('频率（Hz）');
% ylabel('幅值');
% 
% figure(3)
% % plot(freq,new,'-','linewidth',2)
% 
% new_fhz = linspace(0,4000,fft_length/2);
% 
% inter_new = interp1(freq,new,new_fhz);
% plot(new_fhz,inter_new,'-','linewidth',2)
% disp(length(inter_new))
% disp(length(abs_fft(1:fft_length/2)))
% 
% a = reshape(inter_new,1,length(inter_new));
% b = reshape(abs_fft(1:fft_length/2),1,length(inter_new));
% c = a.*b;
% disp(size(c))
% 
% % fft_new = inter_new.*abs_fft(1:fft_length/2);
% % disp(fft_new)
% % 
% figure(4)
% subplot(121);
% plot(new_fhz,c)
% title('经过三角滤波器组后的频谱')
% xlabel('F/hz')
% ylabel('幅值')
% grid on;
% subplot(122);
% plot(new_fhz,abs_fft(1:fft_length/2));
% title('原始频谱')
% xlabel('F/hz')
% ylabel('幅值')
% grid on;




%%
% % 
% % %预加重y=x(i)-0.97*x(i-1)
% % for i=2:3886844
% %     y(i-1)=signal(i)-0.97*signal(i-1);
% % end
% % % disp(size(y))
% % y=y';%对y取转置
% 
% % Wx = fft(signal,512);
% % fhz = fs
% % plot(abs(Wx))
% 
% % x_ = linspace(1,3886844,3886844);
% % y_ = linspace(1,3886843,3886843);
% % % disp(size(x_))
% % % disp(size(y_))
% % figure
% % subplot(1,2,1)
% % 
% % plot(x_,x,'r')
% % title('原始图像')
% % xlabel('样本点')
% % ylabel('幅值')
% % legend()
% % subplot(1,2,2)
% % plot(y_,y,'m')
% % title('预加重')
% % xlabel('样本点')
% % ylabel('幅值')
% % legend()
% 
% % S=enframe(y,1103,662);%分帧,对y进行分帧，x为没有预加重的语音序列
% % %每帧长度为1103个采样点，每帧之间非重叠部分为662个采样点
% % %1103=44100*0.025,   441=44100*0.01    662=1103-441
% % %根据计算，我们可以将108721个数据根据公式662*301+1103=200365
% % %可以将其分为5870帧
% % 
% % 
% % %尝试一下汉明窗a=0.46,得到汉明窗W=(1-a)-a*cos(2*pi*n/N)
% % n=1103;
% % % n = 1:1103;
% % % W = 0.54-0.46*cos((2*pi.*n)/1103);
% % W = hamming(n);
% % 
% % figure(1)
% % plot(W);title('汉明窗');
% % grid on;
% % xlabel('取样点');ylabel('幅值')
% % % 创建汉明窗矩阵C
% % C=zeros(5870,1103);
% % for i=1:5870
% %     C(i,:)=W;
% % end
% % 
% % figure(2)
% % SC=S.*C;
% % subplot(3,1,1);plot(C(1,:),'r');
% % title('汉明窗图像');grid on;%画出第7帧的汉明窗图像
% % hold on
% % subplot(3,1,2);plot(S(1,:),'g');
% % title('原始信号图像');grid on;%画出第7帧的原始信号图像
% % hold on
% % subplot(3,1,3);plot(SC(1,:),'m');
% % title('加了汉明窗的信号图像');grid on;%画出第7帧加了汉明窗的信号图像
% % 
% % 
% % 
% % %对SC的每一帧都进行N=4096的快速傅里叶变换,得到一个301*4096的矩阵
% % F=0;N=4096;
% % for i=1:5870
% %     %对SC作N=4096的FFT变换
% %     D(i,:)=fft(SC(i,:),N);
% %     %以下循环实现求取能量谱密度E
% %     for j=1:N
% %         t=abs(D(i,j));
% %         E(i,j)=(t^2)/N;
% %     end
% %     %获取每一帧的能量总和F(i)
% %     F(i)=sum(D(i,:));
% % end
% 
% 
frameSize=1024;  %帧长
N = frameSize;
inc=512;  %帧移
p = 15;   %梅尔滤波器个数
[x,fs]=audioread('PHONE_001.wav');%读取语音wav文件，本文语音长度3秒，单声道，采样频率16000
% 本文语音信号的fs为16000Hz
N2=length(x);  %语音信号长度

%预加重
for i=2:N2
    y(i)=x(i)-0.97*x(i-1);
end
y=y';%对y取转置
S=enframe(y,frameSize,inc);%分帧,对语音信号x进行分帧，
[a,b]=size(S);  %a为矩阵行数，b为矩阵列数
%创建汉明窗矩阵C
C=zeros(a,b);
ham=hamming(b);
for i=1:a
    C(i,:)=ham;
end
%将汉明窗C和S相乘得SC
SC=S.*C;
F=0;N=frameSize;
for i=1:a
    %对SC作N=1024的FFT变换
    D(i,:)=fft(SC(i,:),N);
    %以下循环实现求取谱线能量
    for j=1:N
        t=abs(D(i,j));
        E(i,j)=(t^2);
    end
end

fl=0;fh=fs/2;%定义频率范围，低频和高频
bl=2595*log10(1+fl/700);%得到梅尔刻度的最小值
bh=2595*log10(1+fh/700);%得到梅尔刻度的最大值
%梅尔坐标范围
% p=26;%滤波器个数
B=bh-bl;%梅尔刻度长度
mm=linspace(0,B,p+2);%规划34个不同的梅尔刻度。但只有32个梅尔滤波器
fm=700*(10.^(mm/2595)-1);%将Mel频率转换为频率
W2=N/2+1;%fs/2内对应的FFT点数,1024个频率分量

k=((N+1)*fm)/fs;%计算34个不同的k值
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
       if n0<=j && j<=n1
           hm(i-1,j)=(j-n0)/(n1-n0);
       elseif n1<=j && j<=n2
           hm(i-1,j)=(n2-j)/(n2-n1);
       end
   end
   %此处求取H1(k)结束。
end

 %梅尔滤波器滤波
 H=E*hm';
 %对H作自然对数运算
 %因为人耳听到的声音与信号本身的大小是幂次方关系，所以要求个对数
 for i=1:a
     for j=1:p
         H(i,j)=log(H(i,j));
     end
 end
 %作离散余弦变换 
Fbank = H';
Fbank1 = dct(Fbank);
mfcc = Fbank1';
plot(mfcc);
imagesc(mfcc');
axis xy;   %y轴从下往上
colorbar;
xlabel('语音分帧帧数');
ylabel('mfcc参数维数');


