function [YM2] = cropMeasurements(YM,G1)
fw = sum(G1(:,end));
w = round(fw/2);
[M,N,K] = size(YM);
[L,L2]=size(G1);
N1 = N-(L-1);
YM2 = zeros(M,N1,K);
 for i=1:K
     YM2(:,:,i) = YM(:,1+(i*fw):N1+(i*fw));
 end
end