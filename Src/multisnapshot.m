function [T] = multisnapshot(RGB,M,N,L,shot,S,nc,G)

T = zeros(M,N,L,shot); % coded aperture for each snapshot
npd = 16;
fltlmbd = 5;
I = sum(RGB,3);
I = RGB(:,:,2);
[sl, sh] = lowpass(I, fltlmbd, npd);
sl = mat2gray(sl);

% Compute the thresholds
%thresh = multithresh(I,nc-1);
thresh = linspace(min(I(:)),max(I(:)),nc);
% Apply the thresholds to obtain segmented image
quantization = imquantize(I,thresh);
%imagesc(quantization)
Q = max(unique(quantization(:)));

for i=1:shot
    for m = 1:M
        for n=1:N
            tm = quantization(m,n);
            tm2 = S(i,tm);
            T(m,n,:,i) = G(:,tm2); 
        end
    end
end

colormap('jet')
for i=1:shot
    str = strcat({'snapshot '} ,num2str(i));
    tm = ceil(shot/2);
    cca = T(:,:,:,i);
    res = 0;
    acum = 0;
    for l=1:L
        acum = 2.^(cca(:,:,l).*l);
        res = res + acum;
    end
    subplot(2,tm,i),imagesc(log(res+1)),title(str);
    %subplot(2,tm,i),imagesc(sum(T(:,:,:,i),3)),title(str);
end

end