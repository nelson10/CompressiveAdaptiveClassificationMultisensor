close all;
clear;
clc;

database = 2;

if(database == 1)
    addpath(genpath('./Real-measurement-pony'));
    I = imread('Groundtruth.png');
elseif(database == 2)
    addpath(genpath('./Real-measurements-penguin'));
    I = imread('groundtruth-penguin.png');
end

subplot(1,3,1),imagesc(I)
[M,N,L] = size(I);
gt = zeros(M,N);

if(database ==1)
    color = [0 0 0;255 255 255;255 0 0;255 102 0;255 255 0;0 128 0;0 0 255;128 0 128];
elseif(database ==2)
    color = [0 0 0;212 0 0;255 127 42;0 0 255;255 255 255];
end
[M1,L1]=size(color);

for m=1:M
    for n=1:N
        temp = double(squeeze(I(m,n,:)));
        for l=1:M1
            if(color(l,:)'==temp)
                gt(m,n)= l-1;
            else
                
            end
        end
    end
end
subplot(1,3,2),imagesc(gt)
colormap(color./255)

if(database ==1)
    load('Barrido1.mat')
    X = Ff{2}(:,:,5:8,:);
    X = sum(X,3)/4;
    x = squeeze(X);
    MS = x;
elseif(database ==2)
    load('Nelson_FullSpectral.mat')
    ind = round(linspace(1,201,12));
    x = dataset(:,:,ind); 
    MS = x;
end

if(database == 1)
    save('Pony-MS-gt.mat','gt','MS')
    load('Pony-MS-gt.mat');
elseif(database == 2)
    gt = uint8(gt);
    save('Penguin-MS-gt.mat','gt','MS')
    load('Penguin-MS-gt.mat');
end
unique(gt)
Img(:,:,3) = sum(MS(:,:,1:4),3);
Img(:,:,2) = sum(MS(:,:,5:8),3);
Img(:,:,1) = sum(MS(:,:,9:12),3);
Img = Img/max(Img(:));
subplot(1,3,3),imagesc(Img)
figure(2)
if(database ==2)
   Img = Img(255:255+541,460:460+541,:);
end
for i=1:max(unique(gt))
    imagesc(Img.*(gt==i-1))
    pause(3.0)
    i
end