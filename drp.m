%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plots for book chapter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Figure 01 - DRP
t = 0:0.01:2;
omega = 2*pi;
phase_s2 = pi/4;
y1 = sin(omega*t);
y2 = sin(omega*t+phase_s2);
h = gca;
peaks_x = [0.25,1.25,9/8];
plot(t,y1,'k','linewidth',2);
h.NextPlot = 'add';
plot(t,y2,'color',ones(1,3)*0.4,'linewidth',2);
plot(peaks_x(1:2),1,'ro','markerface','red');
plot(peaks_x(3),1,'ro','markerface','red');
xlabel('Time[s]');
ylabel('Amplitude');
h_l = legend({'s_1(t)','s_2(t)'});
h_l.FontSize = 12;
h.NextPlot = 'replace';
h.FontName = 'Times New Roman';
line(peaks_x(1)*ones(2,1),[1.2,1.4],'color','black');
line(peaks_x(2)*ones(2,1),[1.2,1.4],'color','black');
line(peaks_x(1:2),[1.3 1.3],'color','black');
text(peaks_x(1),1.5,'t_{1i-1}');
text(peaks_x(2),1.5,'t_{1i}');
line(peaks_x(3)*ones(1,2),0.2+[1.2,1.4],'color','black');
text(peaks_x(3),1.7,'t_{2i}');
h.YLim = [-1 2];
h.YTick = [-1,0,1];
h.XLim = t([1 end]);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 02 - Continuous relative phase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = 0:0.01:4*pi;
y = sin(t);
dy = cos(t);
h = subplot(121);
plot(t,y,'k','linewidth',2);
h.NextPlot = 'add';
plot(t,dy,'color',ones(1,3)*0.4,...
    'linewidth',2);
h.NextPlot = 'replace';
h.XLim = t([1 end]);
xlabel('Time[s]');
ylabel('Amplitude');
h.FontName = 'Times New Roman';
h_l = legend('s','$s\dot(t)$');
h_l.Interpreter = 'latex';
h_l.FontSize = 12;

h = subplot(122);
plot(sin(t),cos(t),'k','linewidth',2);
h.FontName = 'Times New Roman';
h.DataAspectRatio = ones(1,3);
h.XLim = [-1.1 1.1];
h.YLim = h.XLim;
xlabel('s(t)','interpreter','latex');
ylabel('$s\dot(t)$','interpreter','latex');
line([0,sqrt(2)/2*ones(1,2)],...
    [0,sqrt(2)/2*ones(1,2)],'color','black');
line([0,1],[0,0],'color','black');
h.NextPlot = 'add';
plot(cos(pi/4),sin(pi/4),'ro','markerface','red');
arc = 0.5 * [cos(0:0.01:pi/4)',sin(0:0.01:pi/4)'];
plot(arc(:,1),arc(:,2),'color','black');
text(0.6,0.3,'\phi');
n.NextPlot = 'replace';
h.XTick = [-1 0 1];
h.YTick = [-1 0 1];
h.XGrid = 'on'; h.YGrid = 'on';

%% Figure 3 - Hilbert transform fundamentals
T = 0.10;
N = 100;
k = floor(N/2);
omega = 2*pi;
t = 0:T:((N-1)*T);
f = (-k:k-1)*1/(N*T);
y = sin(omega*t);
y_a = hilbert(y);
Y = fftshift(fft(y))/k;
Y_A = fftshift(fft(y_a))/k;

subplot(121);
plot(t,y,'k','linewidth',2);
xlabel('Time[s]');
ylabel('Amplitude');
h = subplot(122);
plot(f,abs(Y),'k','linewidth',2);
h.NextPlot = 'add';
plot(f,abs(Y_A),'color',0.4*ones(1,3),...
    'linewidth',2);
h.NextPlot = 'replace';
h.XLim = [-2 2];
h_l = legend('FT','HT','location','NorthWest');
xlabel('Frequency[Hz]');
ylabel('|Y(\omega)|');
set(get(gcf,'children'),'fontname','Times New Roman');

%% Figure 4 Hilbert transform phase wrapping
t = 0:0.01:4;
A = t;
y = A.* cos(2*pi*t);
A = A(1,end);
h = subplot(131);
plot(t,y,'k','linewidth',2);
h.YLim = [-A*1.1,A*1.1];
h.YTick = [-A,0,A];
h.YTickLabel = {'-A_{max}','0','A_{max}'};
yr = hilbert(y);
title('Signal A(t)*cos(\omega t)');
xlabel('Time[s]');
ylabel('Amplitude');

h = subplot(132);
% instantenous amplitude or envelope
plot(t,abs(yr),'k','linewidth',2);
h.YLim = [0,A*1.1];
h.YTick = [0,A];
h.YTickLabel = {'0','A_{max}'};
title('Instantaneous Amplitude');
xlabel('Times[s]');
ylabel('Amplitude');

h = subplot(133);
% instantenous phase
plot(t,angle(yr),'k','linewidth',2);
line(h.XLim,pi*ones(1,2),'linestyle',':',...
    'color','k');
line(h.XLim,-pi*ones(1,2),'linestyle',':',...
    'color','k')
line(h.XLim,zeros(1,2),'linestyle','-.',...
    'color','k');
h.YTick = [-pi,0,pi];
h.YTickLabel  = {'-\pi','0','\pi'};
title('Instantaneous phase');
xlabel('Time[s]');ylabel('Amplitude');

set(get(gcf,'children'),'fontname','Times New Roman',...
    'xlim',t([1 end]));

%% Figure 5 PCA
T = 0.01;
omega = 2*pi;
noise_amp = 0.2;
t = (0:T:10)';
y1 = 3*t + 3*sin(t);
y2 = 4 + 0.3*y1 + noise_amp*randn(size(t));
y1 = y1 + noise_amp*randn(size(t));
y3 = 10 + 0.5*(t-6).^2 + noise_amp*randn(size(t));
X = [y1,y2,y3];
[coeff,score,latent] = pca(X);

h = subplot(131);
plot(t,y1,'k','linewidth',2);
h.NextPlot = 'add';
plot(t,y2,'color',0.4*ones(1,3),'linewidth',2);
plot(t,y3,'color',0.7*ones(1,3),'linewidth',2);
h.NextPlot = 'replace';
h.XLim = t([1 end]);
legend('Signal 1','Signal 2','Signal 3','location','northwest');
xlabel('Time[s]'); ylabel('Amplitude');

h = subplot(132);
plot(t,score(:,1),'k','linewidth',2);
h.NextPlot = 'add';
plot(t,score(:,2),'color',0.4*ones(1,3),'linewidth',2);
plot(t,score(:,3),'color',0.7*ones(1,3),'linewidth',2);
h.NextPlot = 'replace';
legend('PC1','PC2','PC3','location','northwest');
xlabel('Time[s]'); ylabel('Amplitude');

h = subplot(133);
plot(cumsum(latent / sum(latent))*100,'k-o',...
    'markerface','k','linewidth',2);
line([0.9 3.1],90*ones(1,2),'color','k',...
    'linestyle','--');
h.YLim = [0 110];
xlabel('Principal Components');
ylabel('Cumulated explained variance[%]');
h.XLim = [0.9 3.1];
h.XTick = 1:3;
h.XTickLabel = {'PC1','PC2','PC3'};

set(get(gcf,'children'),'fontname','Times New Roman');
display(latent);

%% cluster phase
T = 0.05;
t = (0:T:10)';
N = numel(t);
K = 5;
omega_sync = 2*pi*ones(1,K);
omega_rnd = unifrnd(0,2*pi,1,K);
phase_sync = unifrnd(-pi/4,pi/4,N,K);
phase_rand = unifrnd(-pi/4,pi/4,N,K);
uniform_noise = unifrnd(0,T,N,K);

theta_sync = t*omega_sync + phase_sync + uniform_noise;
theta_rand = t*omega_rnd + phase_rand + uniform_noise;

theta_sync = mod(theta_sync,2*pi) - pi;
theta_rand = mod(theta_rand,2*pi) - pi;

Y_sync = sin(theta_sync);
Y_rand = sin(theta_rand);

c_p_sync = round(cluster_phase(theta_sync),2);
c_p_rand = round(cluster_phase(theta_rand),2);

colors = (0:K-1)'*ones(1,3)/max(max(K));

h = subplot(121);
h.ColorOrder = colors;
h.NextPlot = 'replaceChildren';
plot(t,Y_sync,'linewidth',1);
title(['Cluster phase = ',num2str(c_p_sync)]);
xlabel('Times[s]');ylabel('Amplitude');

h = subplot(122);
h.ColorOrder = colors;
h.NextPlot = 'replaceChildren';
plot(t,Y_rand,'linewidth',1);
title(['Cluster phase = ',num2str(c_p_rand)]);
xlabel('Times[s]');ylabel('Amplitude');

set(get(gcf,'children'),'fontname','Times New Roman');
