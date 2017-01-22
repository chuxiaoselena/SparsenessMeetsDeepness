function [R,C] = syncRot(T)

[~,L,Q] = proj_deformable_approx(T');
s = sign(L(abs(L)==max(abs(L))));
C = s*L';
R = s*Q';
R(3,:) = cross(R(1,:),R(2,:));

% [R,C] = projectNonrigid(T);
% R(3,:) = cross(R(1,:),R(2,:));
