%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Paper: Adaptive Multisensor Acquisition via Spatial Contextual Information
%   for Compressive Spectral Image Classification
%
%   Fig 2.
%
%   Author:
%   Nelson Eduardo Díaz Díaz,
%   Universidad Industrial de Santander, Bucaramanga, Colombia
%   e-mail: nelson.diaz@saber.uis.edu.co
%   Date Octuber, 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc
close all
addpath(genpath('./Util'));
addpath(genpath('./Util2'));
addpath(genpath('./Data'));
addpath(genpath('./src'));

%% Parameters
Kms = [2 3 4 6]; % vector of number of shots multispectral
Khs = [8 12 16 24]; %number of shots hyperspectral
nm = 4;
shot1 = Kms(nm); %  number of multispectral snapshot
shot2 = Khs(nm); %  number of hiperspectral snapshot

%% Loading data
md = 14; % median filter parameter
adaptive = 1;
dataset2 = 3; % 0 Pavia, 1 Salinas Valley, 2 Indian pines, 3 Hen

if(dataset2 == 0)
    %% Pavia Dataset
    load('PaviaU.mat');
    L1 = 96;
    L2 = 24;
    idx = round(linspace(1,size(paviaU,3),L2));
    Io = mat2gray(paviaU(1:end,1:end,idx));
    MS = Io;
    idx = round(linspace(1,size(paviaU,3),L1));
    Io = paviaU(1:end,1:end,idx);
    for i=1:L1
        HS(:,:,i)=imresize(Io(:,:,i),0.25);
    end
    clear paviaU;
    load('PaviaU_gt.mat');
    %load('../Data/Salinas_gt.mat');
    ground_truth = paviaU_gt(1:end,1:end);
    clear paviaU_gt;
    gt = ground_truth(1:end,1:end);
    R = sum(MS(1:end,1:end,1:8),3);
    G = sum(MS(1:end,1:end,9:16),3);
    B = sum(MS(1:end,1:end,17:24),3);
    RGB(:,:,1) = B;
    RGB(:,:,2) = G;
    RGB(:,:,3) = R;
    RGB = RGB./max(RGB(:));
    RGB1 = imresize(RGB,1);
    RGB2 = imresize(RGB,0.25);
    gt1 = imresize(gt,1,'nearest');
    gt2 = imresize(gt,0.25,'nearest');
    nc = max(gt(:));
elseif(dataset2 ==1)
    %% Salinas Valley
    load('Salinas_corrected.mat');
    L1 = 96;
    L2 = 24;
    idx = round(linspace(1,size(salinas_corrected,3),L2));
    MS = mat2gray(salinas_corrected(1:end,1:end,idx));
    
    idx = round(linspace(1,size(salinas_corrected,3),L1));
    Io = salinas_corrected(1:end,1:end,idx);
    for i=1:L1
        HS(:,:,i)=imresize(Io(:,:,i),0.25);
    end
    clear salinas_corrected;
    load('Salinas_gt');
    gt = salinas_gt(1:end,1:end);
    R = sum(MS(1:end,1:end,1:8),3);
    G = sum(MS(1:end,1:end,9:16),3);
    B = sum(MS(1:end,1:end,17:24),3);
    RGB(:,:,1) = B;
    RGB(:,:,2) = G;
    RGB(:,:,3) = R;
    RGB = RGB./max(RGB(:));
    RGB1 = imresize(RGB,1);
    RGB2 = imresize(RGB,0.25);
    gt1 = imresize(gt,1,'nearest');
    gt2 = imresize(gt,0.25,'nearest');
    nc = max(gt(:));
elseif(dataset2 ==2)
    %% Pavia Dataset
    load('Indian_pines_corrected.mat');
    L1 = 96;
    L2 = 24;
    cube = indian_pines_corrected;
    idx = round(linspace(1,size(cube,3),L2));
    Io = mat2gray(cube(1:end,1:end,idx));
    MS = Io;
    idx = round(linspace(1,size(cube,3),L1));
    Io = cube(1:end,1:end,idx);
    for i=1:L1
        HS(:,:,i)=imresize(Io(:,:,i),0.25);
    end
    clear indian_pines_corrected;
    clear cube;
    load('Indian_pines_gt.mat');
    gt = indian_pines_gt(1:end,1:end);
    R = sum(MS(1:145,1:145,1:8),3);
    G = sum(MS(1:145,1:145,9:16),3);
    B = sum(MS(1:145,1:145,17:24),3);
    RGB(:,:,1) = B;
    RGB(:,:,2) = G;
    RGB(:,:,3) = R;
    RGB = RGB./max(RGB(:));
    RGB1 = imresize(RGB,1);
    RGB2 = imresize(RGB,0.25);
    gt1 = imresize(gt,1,'nearest');
    gt2 = imresize(gt,0.25,'nearest');
    nc = max(gt(:));
elseif(dataset2 == 3)
     load('Hen_FullSpectral.mat');
    %load('Prism');
    %load('NelsonCA.mat');
    L1 = 96;
    L2 = 24;
    cube = dataset(255:255+541,460:460+541,:);
    cube = imresize(cube,0.5);
    idx = round(linspace(1,size(cube,3),L2));
    Io = mat2gray(cube(1:end,1:end,idx));
    MS = Io;
    temp = MS;
    idx = round(linspace(1,size(cube,3),L1));
    Io = cube(1:end,1:end,idx);
    for i=1:L1
        HS(:,:,i)=imresize(Io(:,:,i),0.25);
    end
    clear dataset;
    clear cube;
    R = sum(MS(:,:,1:8),3);
    G = sum(MS(:,:,9:16),3);
    B = sum(MS(:,:,17:24),3);
    RGB(:,:,1) = B;
    RGB(:,:,2) = G;
    RGB(:,:,3) = R;
    RGB = RGB./max(RGB(:));
    imagesc(RGB.^.25)
    RGB1 = imresize(RGB,1);
    RGB2 = imresize(RGB,0.25);
    load('Hen-gt.mat')
    gt1 = imresize(gt,0.5,'nearest');
    gt2 = imresize(gt,0.125,'nearest');
    nc = max(gt(:));
    MS = temp;
else
    %% Pony Dataset
    load('Pony-MS-gt.mat')
    
    R = sum(MS(:,:,1:4),3);
    G = sum(MS(:,:,5:8),3);
    B = sum(MS(:,:,9:12),3);
    RGB(:,:,1) = B;
    RGB(:,:,2) = G;
    RGB(:,:,3) = R;
    RGB = RGB./max(RGB(:));
    RGB = RGB(209:209+511,280:280+511,:);
    gt = gt(209:209+511,280:280+511,:);
    RGB1 = imresize(RGB,0.5,'nearest');
    RGB2 = imresize(RGB,0.125,'nearest');
    %imagesc(RGB)
end

M2 = size(MS,1);
N2 = size(MS,2);
M1 = size(HS,1);
N1 = size(HS,2);
L1 = 96;
L2 = 24;

YH = zeros(M1,N1,shot2);
YM = zeros(M2,N2,shot1);

[Order_fil1,G1] = matchFilter(gt1,MS,shot1);
G1(1:round(size(G1,1)/3),1) = 1; % Capture first band of RGB
[Order_fil2,G2] = matchFilter(gt2,HS,shot2);

figure('Name',"Filters of Multispectral and Hyperspectral Arm")
colormap('jet')
subplot(1,2,1),imagesc(G1),title('Complementary Multispectral filters')
subplot(1,2,2),imagesc(G2),title('Complementary Hyperspectral filters')

if(adaptive == 0)
    T1 = rand(M2,N2,L2,shot1)>0.5;
    T2 = rand(M1,N1,L1,shot2)>0.5;
else
    [T1] = multisnapshot2(RGB1,M2,N2,L2,shot1,Order_fil1,nc,G1);
    [T2] = hypersnapshot(RGB2,M1,N1,L1,shot2,Order_fil2,nc,G2);
end

figure('Name',"Coded apertures in the Multispectral Arm")
showCodedApertures(T1);
figure('Name',"Coded apertures in the Hyperspectral Arm")
showCodedApertures(T2);


size(T1)
size(T2)

% Multispectral snapshots
for i=1:shot1
    t1 = T1(:,:,:,i);
    YM(:,:,i) = sum(t1.*MS,3);
    %if(i==1 && adaptive ==1)
      %  YM(:,:,1)=RGB(:,:,2);
    %end
    if(adaptive==1)
        YM(:,:,i) = medfilt2(YM(:,:,i),[md md]);
    end
end

ym = reshape(YM,[M2*N2,shot1]);

% Hyperspectral snapshots
for i=1:shot2
    t2 = T2(:,:,:,i);
    YH(:,:,i) = sum(t2.*HS,3);
    YH1(:,:,i) = imresize(YH(:,:,i),[M2, N2]);
    if(adaptive==1)
        YH1(:,:,i) = medfilt2(YH1(:,:,i),[md md]);
    end
end

figure('Name',"Example of Compressive Measurements")
subplot(1,2,1),imagesc(YM(:,:,end)),title('Multispectral Compressive Measurement')
subplot(1,2,2),imagesc(YH1(:,:,end)),title('Interpolated Hyperspectral Compressive Measurement')

yh = reshape(YH1,[M2*N2,shot2]);
yt = [ym yh];

training_rate = 0.1;
[training_indexes,test_indexes] = classification_indexes(gt1,training_rate);
T_classes =gt1(training_indexes);

feat_training = yt(training_indexes,:);
feat_test = yt(test_indexes,:);
t = templateSVM('KernelFunction','poly','Standardize',1,'Kernelscale','auto');
MdlSV1 = fitcecoc(feat_training,T_classes,'Learners',t);
yHat = predict(MdlSV1,feat_test);
gtHat = zeros(M2,N2);
gtHat(training_indexes) = T_classes;
gtHat(test_indexes) = yHat;
figure(3)

figure('Name',"Classification Maps")
subplot(1,2,1),imagesc(gt1),title('groundtruth')
subplot(1,2,2),imagesc(gtHat),title('Proposed Algorithm')

[OA1,AA1,kappa1] = compute_accuracy(uint8(gt1(test_indexes)),uint8(gtHat(test_indexes)));
disp("OA= "+num2str(OA1)+" AA= "+num2str(AA1)+" kappa= "+num2str(kappa1))

save('Tensors','T1','T2')