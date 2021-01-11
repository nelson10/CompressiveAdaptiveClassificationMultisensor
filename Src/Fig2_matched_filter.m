nf = 8;         % filter number
L = 16;        % band number
nc = 8;         % number og classes
E = floor(rand(nc,L)*255); 
%g = rand(nb,ng) > 0.5;
G = zeros(L,nc);
wg = 2; % filter width 
G(1:wg,1) = ones(wg,1);
for i=2:nc
    G(:,i) = circshift(G(:,i-1),wg);
end

M = E*G;
[~,ID] = sort(M','descend');
figure(1)
val = zeros(nc,1);
ind = zeros(nc,1);
subplot(1,3,2),imagesc(E),title("E, Endmembers "),xlabel("Band number"),ylabel("Class number")
subplot(1,3,1),imagesc(ID),title("M = E*G matched filter sorted"),xlabel("filter number"),ylabel("Class number")
subplot(1,3,3),imagesc(G),title("G, filters ")
% for i=1:nc
%     [val(i,1),ind(i,1)] = max(max(M));
%     M(:,ind(i,1)) = -realmax;
% end
% ind