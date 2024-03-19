

marker_size = 12;
linewidth1 = 3;


% mm_ts
%{
N_COL = [2 4 6 8 10 12 14 16 18];
time = [0.00020611 0.000417418  0.000637683 0.000856299 0.00109021 0.00128477 0.0015225 0.00174192 0.00196996   ];
bw = [238.475 235.505 231.237 229.602 225.425 229.545 225.987 225.738 224.557  ];
gflops = [119.238 235.505 346.856 459.204 563.562  688.634 790.951  902.951 1010.51 ];


time2 = [ 0.000288693 0.000558963 0.00100263 0.00119125 0.00155539 0.0015869  0.00163948 0.00163435 0.00199126];
bw2 = [170.257 175.869 147.069 165.044 158.006 185.842 209.862 240.596 222.155 ];
gflops2 = [85.1285 175.868 220.603 330.087  395.014 557.525 734.514 962.38 1002.35];

plot( N_COL, bw ,'b-','Marker','pentagram','LineWidth',linewidth1, 'markersize',marker_size )
hold on
plot( N_COL, bw2,'r-','Marker','<','LineWidth',linewidth1, 'markersize',marker_size )


%set(p, 'LineWidth',1, 'markersize',6)
set(gca, 'FontSize', 15)  % Increasing ticks fontsize 


xlabel('Number of columns','Interpreter','latex') 
ylabel('BW[Gb/s]','Interpreter','latex')
title("mm-ts kernel for problem size approx 3000000")
lgnd = legend('my kernel','cublas'); 
set(lgnd, 'Interpreter','latex', 'FontSize',10, 'color','none')
grid on
%}

%#############################################################################################
%mm_tt
%{
N_COL = [2 4 6 8 10 12 14 16 18];
time = [0.000304111 0.000602398 0.000911355 0.00122374  0.00165094 0.00198405 0.00264109  0.00351621 0.00420922];
bw = [533.364 538.52 533.936  530.182  491.239 490.518 429.903 369.038 346.814];
gflops = [266.682  538.52 800.903 1060.36 1228.1 1471.55 1504.66 1476.15 1560.66 ];


time2 = [0.0054724 0.00568016 0.00590421 0.00613327 0.00647135 0.00669459 0.00692208 0.00711768 0.00735657   ];
bw2 = [ 29.6399 57.1116 82.4166 105.785 125.323 145.373 164.028 182.308 198.437];
gflops2 = [14.82 57.1116  123.625 211.57 313.307 436.118 574.096 729.233 892.966 ];
figure(1)
plot( N_COL, bw ,'b-','Marker','pentagram','LineWidth',linewidth1, 'markersize',marker_size )
hold on
plot( N_COL, bw2,'r-','Marker','<','LineWidth',linewidth1, 'markersize',marker_size )


set(gca, 'FontSize', 15)  % Increasing ticks fontsize 

xlabel('Number of columns','Interpreter','latex') 
ylabel('BW[Gb/s]','Interpreter','latex')
title("mm-tt kernel for problem size approx 10000000")
lgnd = legend('my kernel','cublas'); 
set(lgnd, 'Interpreter','latex', 'FontSize',10, 'color','none')
grid on
%}
%#############################################################################################
%mm_tt2
%{
N_COL = [2 4 6 8 10 12 14 16 18];
time = [0.000597606 0.0012171 0.00186971 0.00246859 0.00316379 0.00390869 0.00540508 0.00663088 0.00823899];
bw = [542.838 533.075 520.513 525.649 512.682 497.973 420.128 391.385 354.368];
gflops = [271.419 533.075 780.77 1051.3 1281.7 1493.92 1470.45 1565.54 1594.65];


time2 = [ 0.0112732  0.0119653 0.0126655 0.0134042 0.0142175 0.014225 0.0158936 0.0169122 0.0181004];
bw2 = [28.7764 54.224 76.8393 96.8063 114.086 114.026 142.877 153.453 161.302];
gflops2 = [ 14.3882 54.224 115.259 193.613 285.214 285.064 500.068 613.81 725.86];
figure(2)
plot( N_COL, bw ,'b-','Marker','pentagram','LineWidth',linewidth1, 'markersize',marker_size )
hold on
plot( N_COL, bw2,'r-','Marker','<','LineWidth',linewidth1, 'markersize',marker_size )


set(gca, 'FontSize', 15)  % Increasing ticks fontsize 

xlabel('Number of columns','Interpreter','latex') 
ylabel('GFLOPS','Interpreter','latex')

title("mm-tt2 kernel for problem size approx 10000000")

lgnd = legend('my kernel','cublas'); 
set(lgnd, 'Interpreter','latex', 'FontSize',10, 'color','none')
grid on 
%}
%#############################################################################################
%spmm spmv
%{
N_COL = [2 4 6 8 10 12 14 16 18];
time = [ 0.00450626 0.00617089 0.00791544 0.00969507 0.0114425 0.0131959 0.0149673 0.0167121 0.0184522];
bw = [264.239 257.279 250.719 245.636 242.811 240.627 238.666 237.499 236.612];
gflops = [88.0797 128.639 150.431  163.757 173.436  180.47 185.629 189.999 193.592];




plot( N_COL, gflops ,'b-','Marker','pentagram','LineWidth',linewidth1, 'markersize',marker_size )


set(gca, 'FontSize', 15)  % Increasing ticks fontsize 

xlabel('Number of columns','Interpreter','latex') 
ylabel('GFLOPS','Interpreter','latex')
title("spmm kernel for problem size approx 24000000")
lgnd = legend('my kernel'); 
set(lgnd, 'Interpreter','latex', 'FontSize',10, 'color','none')
grid on 
%}
%#############################################################################################
%sqrtm
%{
title("sqrtm kernel")
N_COL = [2 4 6 8 10 12 14 16 18];
time = [3.87581e-06 1.27183e-05 2.38117e-05 4.11426e-05 5.37104e-05 6.28972e-05 7.58798e-05 0.000115611 0.000131374];
time2 = [2.51966e-05 2.18486e-05 2.23428e-05 2.28633e-05 2.98761e-05 3.93657e-05 6.17708e-05 7.65662e-05 8.42831e-05];

plot( N_COL, time ,'b-','Marker','pentagram','LineWidth',linewidth1, 'markersize',marker_size )
hold on
plot( N_COL,time2,'r-','Marker','<','LineWidth',linewidth1, 'markersize',marker_size )


set(gca, 'FontSize', 15)  % Increasing ticks fontsize 

xlabel('Number of columns','Interpreter','latex') 
ylabel('Time[s]','Interpreter','latex')

lgnd = legend('my kernel','based on cusolver syevjBatched'); 
set(lgnd, 'Interpreter','latex', 'FontSize',10, 'color','none')
grid on 

%}
%######################################################################################################
%lanczos
%
N_COL = [2 4 6 8 10 12 14 16 18];
time = [0.0285895 0.0525336 0.080039 0.109402 0.12206 0.142111 0.165825 0.197981 0.254693 ];
bw = [251.344 243.179 235.402 238.129 240.767 246.701 245.331 235.091 208.966];
gflops = [97.7412 182.38 238.812 319.92 439.237 538.261 622.959 680.972 679.971 ];

time2 = [2.51966e-05 2.18486e-05 2.23428e-05 2.28633e-05 2.98761e-05 3.93657e-05 6.17708e-05 7.65662e-05 8.42831e-05];

%plot( N_COL, bw ,'b-','Marker','pentagram','LineWidth',linewidth1, 'markersize',marker_size )

cal = (1./((time./N_COL)./(0.0166125)) - 1)*100;
%cal = (2*time2)./time *100;
plot( N_COL, cal,'r-','Marker','<','LineWidth',linewidth1, 'markersize',marker_size )


set(gca, 'FontSize', 15)  % Increasing ticks fontsize 

xlabel('Number of columns','Interpreter','latex') 
ylabel('percentage speedup[%]','Interpreter','latex')
title(" speed up of a single vector in block lanczos aginst vector lanczos  ")
lgnd = legend('percentage speadup');
set(lgnd, 'Interpreter','latex', 'FontSize',12, 'color','none')
grid on 
%}

%######################################################################################################
%convergence
%{
m = [1 2 3 4 5 6 7 8 9 10 11 12];
error = [ 0.0552507  0.000769318 0.000153389  6.64577e-07  3.88449e-07 1.05489e-09 2.85955e-09 1.83726e-09 1.8275e-09 1.8275e-09 1.83022e-09 1.83025e-09];


plot( m, error ,'b-','Marker','pentagram','LineWidth',linewidth1, 'markersize',marker_size )

%plot( N_COL,bw2,'r-','Marker','<','LineWidth',linewidth1, 'markersize',marker_size )


set(gca, 'FontSize', 15)  % Increasing ticks fontsize 
set(gca, 'YScale', 'log')
xlabel('Number of iterations','Interpreter','latex') 
ylabel('Relative error','Interpreter','latex')
title(" Convergence for problem size 252 aginst fdtd with 1000000 steps  ")
lgnd = legend('relative error'); 
set(lgnd, 'Interpreter','latex', 'FontSize',10, 'color','none')
grid on 
%}
%######################################################################################################
%{
v = readmatrix("file_VL.csv");
B = readmatrix("file_BL.csv");
size1 = v(:,1);
time1 = v(:,2);
bw1 = v(:,3);
gflops1 = v(:,4);

size2 = B(:,2);
time2 = B(:,3);
bw2 = B(:,4);
gflops2 = B(:,5);

%cal = (1./((time2(1:18)./18)./(time1(1:18))) - 1)*100;
plot( size2(1:18), gflops2(1:18) ,'b-','Marker','pentagram','LineWidth',linewidth1, 'markersize',marker_size )
%hold on 
%plot( size2(1:18), time2(1:18),'r-','Marker','<','LineWidth',linewidth1, 'markersize',marker_size )


set(gca, 'FontSize', 15)  % Increasing ticks fontsize 
set(gca, 'XScale', 'log')
xlabel('Problem size','Interpreter','latex') 
ylabel('GFLOPS','Interpreter','latex')
title(" GFLOPS block lanczos vs problem size ")
lgnd = legend( 'GFLOPS block lanczos'); 
set(lgnd, 'Interpreter','latex', 'FontSize',10, 'color','none')
grid on 


%}






