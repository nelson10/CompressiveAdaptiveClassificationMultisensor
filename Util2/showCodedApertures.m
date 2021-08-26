function showCodedApertures(T)
[M,N,L,shot]=size(T);
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
