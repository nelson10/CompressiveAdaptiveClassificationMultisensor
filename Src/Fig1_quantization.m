close all;
I = 1:256;
delta = 32;
Q = delta*floor(I/delta + 0.5);
plot(I,Q)
title("Quantizer");
xlabel("Intensity values");
ylabel("Quantized values");