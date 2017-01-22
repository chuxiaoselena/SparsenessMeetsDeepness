function [dist,S2] = compareStructs(S1,S2,sep)

if nargin < 3
    sep = false;
end

% compare two structures and align S2 to S1 with similarity transformation

T1 = mean(S1,2);
S1 = bsxfun(@minus,S1,T1);
S2 = bsxfun(@minus,S2,mean(S2,2));

[f,p] = size(S1);
f = f/3;
dist = zeros(f,1);

Y = findRotation(S1,S2);
for i = 1:f
    A = S1(3*i-2:3*i,:);
    B = S2(3*i-2:3*i,:);
    if sep
        Y = findRotation(A,B);
    end
    % rotate B
    B = Y*B;
    % scale B
    w = trace(A'*B)/trace(B'*B);
    B = w*B;
    % output
    dist(i) = sum(sqrt(sum((A-B).^2,1)))/p; % average distance
    S2(3*i-2:3*i,:) = B;
end

S2 = bsxfun(@plus,S2,T1);

end

function R = findRotation(S1,S2)
[F,P] = size(S1);
F = F/3;
S1 = reshape(S1,3,F*P);
S2 = reshape(S2,3,F*P);
R = S1*S2';
[U,~,V] = svd(R);
R = U*V';
% R = U*diag([1 1 det(R)])*V';
end
