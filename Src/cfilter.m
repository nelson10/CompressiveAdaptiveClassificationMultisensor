function G = cfilter(L,nf)

if(L >= nf)
    w = round(L/nf); % filter width
else
    disp('L is smaller than shots')
end

G = zeros(L,nf);
G(1:w,1) = ones(w,1);

for i=2:nf
    G(:,i) = circshift(G(:,i-1),w);
end

figure(1)
colormap('jet')
subplot(1,2,1),imagesc(G),title('Complementary filters')
end