clear all;
nf = 8;         % filter number
L = 96;        % band number
M = 256;
N = 256;

addpath(genpath('./Util'));
addpath(genpath('./Data'));
addpath(genpath('./src'));

temp = zeros(L,M*N);

load('paviaU');
id = round(linspace(1,103,L));
paviaUD = mat2gray(paviaU(end-255:end,1:256,id));
%clear paviaU;
load('PaviaU_gt.mat');
ground_truth = paviaU_gt(end-255:end,1:256);
clear paviaU_gt;
c = unique(ground_truth(:));
nc = length(c)-1; % number og classes
E = zeros(L,nc);
F = reshape(paviaUD,[L M*N]);
for i=2:nc+1
    T = (ground_truth==c(i));
    ind = T(:)';
    E(:,i-1) = sum(F(:,ind),2)/sum(ind(:));  
    plot(E(:,i-1))
    hold on
end

%E = floor(rand(nc,L)*255);
%g = rand(nb,ng) > 0.5;
B = zeros(L,nc);
wg = L/nc; % filter width
B(1:wg,1) = ones(wg,1);
for i=2:nc
    B(:,i) = circshift(B(:,i-1),wg);
end

E = E';
M = E*B;
[~,ID] = sort(M','descend');
figure(1)
val = zeros(nc,1);
ind = zeros(nc,1);
subplot(1,4,3),imagesc(E),title("E, Endmembers "),xlabel("Number of bands"),ylabel("Class number")
ax = gca;
ax.FontSize = 18;
subplot(1,4,1),imagesc(ID),title("sort(M') matched filter sorted"),xlabel("Class number"),ylabel("Class number")
ax = gca;
ax.FontSize = 18;
subplot(1,4,2),imagesc(M),title("M = E*B matched filter"),xlabel("Shot number"),ylabel("Class number")
ax = gca;
ax.FontSize = 18;
subplot(1,4,4),imagesc(B),title("B, filters "),xlabel("Shot number"),ylabel("Number of bands")
% Get handle to current axes.
ax = gca;
ax.FontSize = 18;

%outputBaseFileName = 'image.PNG';
%imwrite(originalImage, outputBaseFileName);

% for i=1:nc
%     [val(i,1),ind(i,1)] = max(max(M));
%     M(:,ind(i,1)) = -realmax;
% end
% ind




