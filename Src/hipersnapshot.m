function [T] = hipersnapshot(RGB,M,N,L,shot,Order_fil,nc,G)

T = zeros(M,N,L,shot); % coded aperture for each snapshot
npd = 16;
fltlmbd = 5;
I = RGB(:,:,2);
I = mat2gray(I);
[sl, sh] = lowpass(I, fltlmbd, npd);
sh = mat2gray(sh);

% Compute the thresholds
thresh = multithresh(sh,nc-1);
%thresh = linspace(min(I(:)),max(I(:)),nc);
% Apply the thresholds to obtain segmented image
quantization = imquantize(sh,thresh);
%imagesc(quantization)
%Q = max(unique(quantization(:)));

for i=1:shot
    for m = 1:M
        for n=1:N
            tm = quantization(m,n);
            tm2 = Order_fil(i,tm);
            T(m,n,:,i) = G(:,tm2); 
        end
    end
end
end