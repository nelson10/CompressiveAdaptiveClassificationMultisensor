function [T] = multisnapshot2(RGB,M,N,L,shot,S,nc,G)

    T = zeros(M,N,L,shot); % coded aperture for each snapshot
    npd = 16;
    fltlmbd = 5;
    I = RGB(:,:,1); % This is equivalent to cap
    %I = I./max(I(:));
    I = mat2gray(I);
    [sl, sh] = lowpass(I, fltlmbd, npd);
    sh =mat2gray(sh);
    
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
                tm2 = S(i,tm);
                T(m,n,:,i) = G(:,tm2);
            end
        end
    end
end