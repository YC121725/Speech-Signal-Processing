tic;
clear; close all; clc;

%% Ԥ���ء���֡���Ӵ�
frameSize = 200;  % ֡��
inc = 80;  % ֡���
[x , fs] = audioread('PHONE_001.wav');  % ��ȡ PHONE_001.wav �ļ�
x = x(1:10000);
x = x';

N = length(x);
x = double(x);
y = zeros(1,length(x));

% Ԥ���� a=0.97
for i = 2:N
    y(i) = x(i)-0.97*x(i-1);
end

figure(1)
subplot(211);
xx = x(1000:1200);
yy = y(1000:1200);
plot(xx,'g'); hold on;
plot(yy,'r');
legend('һ֡ԭʼ�ź�','һ֡�����ź�');
axis([0 200 -0.003 0.003])   
subplot(212);
plot(abs(fft(xx,512)),'g'); hold on;
plot(abs(fft(yy,512)),'r'); hold on;
legend('ԭʼ�ź�Ƶ��','�����ź�Ƶ��');
axis([0 512 0 0.05])   

S = enframe(y,frameSize,inc);  % ��֡,��x���з�֡
[a, b] = size(S);

% ������a=0.46,�õ�������W=(1-a)-a*cos(2*pi*n/N)
n = 1:b;
W = 0.54-0.46*cos((2*pi.*n)/b);

% ��������������C
C = zeros(a,b);
ham = hamming(b);
for i = 1:a
    C(i,:) = ham;
end

% ��������C��S��˵�SC
SC = S.*C;

figure(2)
subplot(3,1,1); plot(C(10,:),'r');
title('������ͼ��'); grid on;  % ������10֡�ĺ�����ͼ��
subplot(3,1,2);plot(S(10,:),'g');
title('ԭʼ�ź�ͼ��'); grid on;  % ������10֡��ԭʼ�ź�ͼ��
subplot(3,1,3); plot(SC(10,:),'m');
title('���˺��������ź�ͼ��'); grid on;  % ������10֡���˺��������ź�ͼ��

figure(3)
plot(S(10,:),'g');hold on;
plot(SC(10,:),'m');
legend('һ��ԭʼ�ź�','������ź�');

F = 0;
N = 512;
for i=1:a
    %��SC�� N=512 ��FFT�任
    D(i,:)=fft(SC(i,:),N);
    %����ѭ��ʵ����ȡ�������ܶ�E
    for j=1:N
        t=abs(D(i,j));
        E(i,j)=(t^2)/N;
    end
    %��ȡÿһ֡�������ܺ�F(i)
    F(i)=sum(E(i,:));
    i/a;
end

%��Ƶ��ת��Ϊ÷��Ƶ��
%÷��Ƶ��ת������ͼ��
mel = zeros(1,fs);
for i=1:fs
    mel(i)=2595*log10(1+i/700);
end
fl = 60;  %����Ƶ�ʷ�Χ����Ƶ
fh = 3400;%����Ƶ�ʷ�Χ����Ƶ
bl = 2595*log10(1+fl/700);%�õ�÷���̶ȵ���Сֵ
bh = 2595*log10(1+fh/700);%�õ�÷���̶ȵ����ֵ

%÷�����귶Χ
p = 15;%�˲�������
mm = linspace(bl,bh,p+2);%�滮17����ͬ��÷���̶�
fm = 700*(10.^(mm/2595)-1);%��MelƵ��ת��ΪƵ��
figure(4)
plot(mel,'linewidth',1.5);grid on;hold on;
plot(fm,mm,'o','color','r');
title('÷��Ƶ��ת��');xlabel('Ƶ��');ylabel('÷��Ƶ��');

W2 = N/2+1;%fs/2�ڶ�Ӧ��FFT����,257��Ƶ�ʷ���
k = ((N+1)*fm)/fs%����17����ͬ��kֵ
hm = zeros(15,N);%����hm����
df = fs/N;
freq = (0:N-1)*df;%����Ƶ��ֵ

%����÷���˲���
for i = 2:16
    %ȡ��������ȡ����16��k�еĵ�2-15��������1��17
    n0 = floor(k(i-1));
    n1 = floor(k(i));
    n2 = floor(k(i+1));
    %k(i)�ֱ�������ÿ��÷��ֵ���µķ�Χ�ڵ�ӳ�䣬��������
    for j = 1:N
       if n0<=j & j<=n1
           hm(i-1,j)=(j-n0)/(n1-n0);
       elseif n1<=j & j<=n2
           hm(i-1,j)=(n2-j)/(n2-n1);
       end
    end 
end

%��ͼ,��ÿ����ɫ��ʾ��һ��
c = colormap(lines(15));%����15����ͬ��ɫ������
figure(5)
for i = 1:15
    plot(freq,hm(i,:),'--','color',c(i,:),'linewidth',2);%��ʼѭ������ÿ��÷���˲���
    hold on
end
 grid on;%��ʾ����
 axis([0 4000 0 1]);%������ʾ��Χ
 
 %�õ��������������ĺ�
 H = E*hm';
 
 %��H����Ȼ��������
 %��Ϊ�˶��������������źű���Ĵ�С���ݴη���ϵ������Ҫ�������
 for i = 1:a
     for j = 1:15
         H(i,j) = log(H(i,j));
     end
 end

%����ɢ���ұ任DCT   ����MFCC����
mfcc = zeros(a,15);
for i = 1:a
    for j = 1:15
        %����ȡÿһ֡�������ܺ�
        sum1 = 0;
        %����ɢ���ұ任
        for p = 1:15
            sum1 = sum1+H(i,p)*cos((pi*j)*(2*p-1)/(2*15));
        end
        mfcc(i,j) = ((2/15)^0.5)*sum1;     %MFCC����
        %�����ɢ���ұ任
    end    
end
 
 
% ��ּ���
% �����������㣬��һ��������ߴ���
% ��Ϊ�󲿷ֵ��ź�����һ�㼯���ڱ任��ĵ�Ƶ�������Զ�ÿһֻ֡ȡǰ13�����ݾͺ���
J = mfcc(:,(1:13));
K = zeros(1,13);
for i=1:13
    K(i)=1+(22/2)*sin(pi*i/22);
end
K = K/max(K);
 
% �õ���ά����m,����mfcc��ԭʼ����;
L = zeros(a,13);
for i = 1:a
    for j = 1:13
        L(i,j) = J(i,j)*K(j);
    end
end

feat = L;
% ��������ȡmfcc���������ײ�ֲ��� 48584x13
% ��������ȡ�ڶ��飨һ�ײ��ϵ����48584x13����Ҳ��mfcc�����ĵڶ������
% dtfeat = 0;
dtfeat = zeros(size(L));%Ĭ�ϳ�ʼ��
for i=3:a-2
    dtfeat(i,:) = -2*feat(i-2,:)-feat(i-1,:)+feat(i+1,:)+2*feat(i+2,:); 
end
dtfeat = dtfeat/10;

% ��ȡ���ײ��ϵ��,mfcc�����ĵ��������
% ���ײ��ϵ�����Ƕ�ǰ�������һ�ײ��ϵ��dtfeat�ٴν��в�����
% dttfeat = 0;
dttfeat = zeros(size(dtfeat));%Ĭ�ϳ�ʼ��
for i=3:a-2
    dttfeat(i,:) = -2*dtfeat(i-2,:)-dtfeat(i-1,:)+dtfeat(i+1,:)+2*dtfeat(i+2,:); 
end
dttfeat = dttfeat/10;

% ��ȡ���ײ��ϵ��,mfcc�����ĵ��������
% ���ײ��ϵ�����Ƕ�ǰ�������һ�ײ��ϵ��dtfeat�ٴν��в�����
% dtttfeat = 0;
dtttfeat = zeros(size(dtfeat));%Ĭ�ϳ�ʼ��
for i=3:a-2
    dtttfeat(i,:) = -2*dttfeat(i-2,:)-dttfeat(i-1,:)+dttfeat(i+1,:)+2*dttfeat(i+2,:); 
end
dtttfeat = dtttfeat/10;

%���õ���mfcc����������feat��dtfeat��dttfeatƴ�ӵ�һ��
%�õ�����mfccϵ��48584x52
% mfcc_final = 0;
mfcc_final = [feat,dtfeat,dttfeat,dtttfeat];%ƴ�����
disp(size(feat))
disp(size(dtfeat))
disp(size(mfcc_final))
% % ��ȥ��β��֡����Ϊ����֡�Ĳ�ֲ���Ϊ0
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
% title('MFCC(1-15)ά');
% 
% mfcc2 = mfcc';
% mfcc2 = mfcc2(1:14,:);
% t2 = linspace(0,size(mfcc2,2)*80/8000,485);
% f2 = linspace(1,size(mfcc2,1)-1,14);
% figure(7);
% imagesc(t2, f2, 20*log10((abs(mfcc2))));xlabel('s'); ylabel('db');
% colorbar;
% title('MFCC(1-14)ά');
% 
% mfcc_final1 = mfcc_final';
% mfcc_final1 = mfcc_final1(14:52,:);
% t3 = linspace(0,size(mfcc_final,1)*80/8000,485);
% f3 = linspace(14,size(mfcc_final,2),39);
% figure(8);
% imagesc(t3, f3, 20*log10((abs(mfcc_final1))));xlabel('s'); ylabel('db');
% colorbar;
% title('MFCC���(14-52)ά');
% 
% cmvn1 = cmvn';
% cmvn1 = cmvn1(14:52,:);
% t4 = linspace(0,size(cmvn,1)*80/8000,485);
% f4 = linspace(14,size(cmvn,2),39);
% figure(9);
% imagesc(t4, f4, 20*log10((abs(cmvn1))));xlabel('s'); ylabel('db');
% colorbar;
% title('MFCC���-CMVN(14-52)ά');
% 
% toc;