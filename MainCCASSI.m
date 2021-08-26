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
nm = 1;
shot1 = Kms(nm); %  number of multispectral snapshot
shot2 = Khs(nm); %  number of hyperspectral snapshot

%% Loading data
md = 12; % median filter parameter
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
    load('Hen-MS-gt.mat')
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

M2 = size(MS,1); % spatial dimension multispectral
N2 = size(MS,2); % spatial dimension multispectral
M1 = size(HS,1); % spatial dimension hyperspectral
N1 = size(HS,2); % spatial dimension hyperspectral
L1 = 96; % spectral dimension hyperspectral
L2 = 24; % spectral dimension multispectral

YH = zeros(M1,N1+L1-1,shot2);
YM = zeros(M2,N2+L2-1,shot1);

[Order_fil1,G1] = matchFilter(gt1,MS,shot1);
G1(1:round(size(G1,1)/3),1) = 1;
[Order_fil2,G2] = matchFilter(gt2,HS,shot2);

if(adaptive == 0)
    T1 = rand(M2,N2,L2,shot1)>0.5;
    T2 = rand(M1,N1,L1,shot2)>0.5;
else
    [T1] = multisnapshot2(RGB1,M2,N2,L2,shot1,Order_fil1,nc,G1);
    [T2] = hypersnapshot(RGB2,M1,N1,L1,shot2,Order_fil2,nc,G2);
end

figure('Name',"Coded apertures in the Multispectral Arm")
showCodedApertures(T1);
figure('Name',"Coded apertures in the Hiperspectral Arm")
showCodedApertures(T2);

size(T1)
size(T2)

% Multispectral snapshots
for i=1:shot1
    t1 = T1(:,:,:,i);
    for j=1:L2
        YM(:,1+(j-1):N2+(j-1),i) = YM(:,1+(j-1):N2+(j-1),i) + (t1(:,:,j).*MS(:,:,j));
    end
    if(adaptive==1)
        YM(:,:,i) = medfilt2(YM(:,:,i),[md md]);
    end
end

[YM2] = cropMeasurements(YM,G1); % Cropping Multispectral Measurements
ym = reshape(YM2,[M2*N2,shot1]);

% Hyperspectral snapshots
for i=1:shot2
    t2 = T2(:,:,:,i);
    for j=1:L1
        YH(:,1+(j-1):N1+(j-1),i) = YH(:,1+(j-1):N1+(j-1),i) + (t2(:,:,j).*HS(:,:,j));
    end
    YH1(:,:,i) = imresize(YH(:,:,i),[M2, (N2+L1-1)]);
    if(adaptive==1)
        YH1(:,:,i) = medfilt2(YH1(:,:,i),[md md]);
    end
end

[YH2] = cropMeasurements(YH1,G2); % Cropping Hyperspectral Measurements
yh = reshape(YH2,[M2*N2,shot2]);
yt = [ym yh]; 

figure(4)
subplot(2,2,1),imagesc(YM(:,:,1))
subplot(2,2,2),imagesc(YM2(:,:,1))
subplot(2,2,3),imagesc(YH1(:,:,1))
subplot(2,2,4),imagesc(YH2(:,:,1))


%% Classification process using SVM with Polynomial Kernel Function
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
figure('Name',"Classification Maps")

subplot(1,2,1),imagesc(gt1),title('groundtruth')
subplot(1,2,2),imagesc(gtHat),title('Proposed Algorithm')

[OA1,AA1,kappa1] = compute_accuracy(uint8(gt1(test_indexes)),uint8(gtHat(test_indexes)));
disp("OA= "+num2str(OA1)+" AA= "+num2str(AA1)+" kappa= "+num2str(kappa1))

save('Tensors','T1','T2')