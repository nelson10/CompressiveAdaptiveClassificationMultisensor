function [ID,G] = matchFilter(gt,MS,shot)
nc = max(gt(:));
[M,N,L] = size(MS);
I = reshape(MS,[L M*N]);
nf = shot;
% the size of G is L x nf
%G is L x nf
%E is nc x L
%M is nc x nf
% M = E*G;
G = cfilter(L,nf); % filter per snapshot
E = zeros(nc,L);

for i=1:nc
    T = (gt==i);
    E(i,:) = mean(I(:,T(:)),2);
end

M = E*G;
[~,ID] = sort(M','descend');
end