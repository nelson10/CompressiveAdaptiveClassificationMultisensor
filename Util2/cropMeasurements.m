function [YM2] = cropMeasurements(YM,G1)
fw = sum(G1(:,end));
w = round(fw/2);
[M,N,L] = size(YM);
%N1 = N-M+1;
YM2 = zeros(M,M,L);
 for i=1:L
     YM2(:,:,i) = YM(:,1+(i*fw):M+(i*fw));
 end
end