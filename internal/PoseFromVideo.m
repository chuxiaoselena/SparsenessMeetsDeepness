function output = PoseFromVideo(varargin)

% general syntex: 

% (1) if 2D pose W_gt is given
% output = PoseFromVideo('W_gt',W_gt,'dict',dict)
% where "W_gt" is a 2D array: 2*#frames by #joints
% every two rows of W_gt store xy joint locations of a 2D pose

% (2) if only joint heatmaps are given 
% output = PoseFromVideo('heatmap',heatmap,'dict',dict)
% where "heatmap" is a 4D array: heigth-by-width-by-#joints-by-#frames

% "dict" is a structure stores the shape dictionary 
% The recoverd 3D poses are stored in "output.S_final"
% which is a 2D array: 3*#frames by #joints
% every three rows of output.S_final store xyz joint locations of a 2D pose


heatmap = []; % heatmap from CNNs (4D array)
W_gt = []; % groundtruth 2D pose
S_gt = []; % groundtruth 3D pose
dict = []; % 3D pose dictionary
alpha = 0.5; % weight of sparsity
beta = 20; % weight of smoothness of coefficient
gamma = 2; % weight of smoothness of rotation
sigma = 0.5; % hyperparameter in (4), gamma = 1/sigma^2
tol = 1e-4; % criterion of convergence
MaxIterEM = 10; % maximum number of EM iterations
MaxIterAltern = 10; % maximum number of inner iterations
InitialMethod = 'convex+robust+refine'; % initialization method
FilterRotation = false; % filtering rotation in initialization or not
filterSize = 5; % filter window size
verb = true; % display information or not
output = [];

ivargin = 1;
while ivargin <= length(varargin)
    switch lower(varargin{ivargin})
        case 'heatmap'
            ivargin = ivargin + 1;
            heatmap = varargin{ivargin};
        case 'w_gt'
            ivargin = ivargin + 1;
            W_gt = varargin{ivargin};
        case 'dict'
            ivargin = ivargin + 1;
            dict = varargin{ivargin};
        case 'alpha'
            ivargin = ivargin + 1;
            alpha = varargin{ivargin};
        case 'beta'
            ivargin = ivargin + 1;
            beta = varargin{ivargin};
        case 'gamma'
            ivargin = ivargin + 1;
            gamma = varargin{ivargin};
        case 'sigma'
            ivargin = ivargin + 1;
            sigma = varargin{ivargin};
        case 'filtersize'
            ivargin = ivargin + 1;
            filterSize = varargin{ivargin};
        case 'tol'
            ivargin = ivargin + 1;
            tol = varargin{ivargin};
        case 'initialmethod'
            ivargin = ivargin + 1;
            InitialMethod = varargin{ivargin};
        case 'filterrotation'
            ivargin = ivargin + 1;
            FilterRotation = varargin{ivargin};
        case 'maxiterem'
            ivargin = ivargin + 1;
            MaxIterEM = varargin{ivargin};
        case 'maxiteraltern'
            ivargin = ivargin + 1;
            MaxIterAltern = varargin{ivargin};
        case 's_gt'
            ivargin = ivargin + 1;
            S_gt = varargin{ivargin};
        case 'init'
            ivargin = ivargin + 1;
            output = varargin{ivargin};
        case 'verb'
            ivargin = ivargin + 1;
            verb = varargin{ivargin};
        otherwise
            fprintf('Unknown option ''%s'' is ignored !!!\n',...
                varargin{ivargin});
    end
    ivargin = ivargin + 1;
end

% output structure
if isempty(output)
    output = struct(...
        'S_init',[],...  % initial 3D pose
        'R_init',[],...  % initial camera rotations
        'C_init',[],...  % initial coefficients
        'T_init',[],...  % initial translation
        'S_final',[],... % optimized 3D pose
        'R_final',[],... % optimized camera rotations
        'C_final',[],... % optimized coefficients
        'T_final',[]);   % optimized translation
end

heatmap(heatmap<0) = 0;
size_heatmap = size(heatmap);

if isempty(W_gt) && ~isempty(heatmap)
    EM = true;
    [X,Y] = meshgrid(1:size_heatmap(2),1:size_heatmap(1));
    xy = [X(:),Y(:)]';
    W_init = findWmax(heatmap);
    % normalize data scale for convenience of paramter setting
    size_metric = 6;
    scale = size_heatmap(1)/size_metric;
    xy = xy / scale;
    W_init = W_init / scale;
elseif ~isempty(W_gt)
    EM = false;
    W_init = W_gt;
    % normalize data scale for convenience of paramter setting
    scale = mean(std(W_gt,1,2));
    size_metric = size_heatmap(1)/scale;
    W_init = W_init / scale;
else
    fprintf('No input data!\n');
    return
end

nFrame = size(W_init,1)/2;
nJoint = size(W_init,2);

%% single frame initialization
if isempty(output.R_init)
    parfor i = 1:nFrame
        fprintf('Single frame initialization, frame %d\n',i);
        [S,info] = ssr2D3D_wrapper(W_init(2*i-1:2*i,:),dict.B,InitialMethod);
        S_init{i,1} = S;
        C_init{i,1} = info.C;
        R_init{i,1} = info.R(1:2,:);
        T_init{i,1} = info.T;
    end
    S_init = cell2mat(S_init);
    R_init = cell2mat(R_init);
    C_init = cell2mat(C_init);
    T_init = cell2mat(T_init);
else
    S_init = output.S_init(1:3*nFrame,:);
    R_init = output.R_init(1:2*nFrame,:);
    C_init = output.C_init(1:nFrame,:);
    T_init = output.T_init(1:2*nFrame,:);
end

S = S_init;
C = C_init;
R = R_init;
T = T_init;
W = W_init;

if FilterRotation
    fprintf('Median filtering of rotations ... \n');
    R = medfiltRotations(R,filterSize);
end

if ~isempty(S_gt)
    e = compareStructs(S_gt,S,'true');
    e = mean(e);
    fprintf('Initialization error = %.2f\n',e);
end

%%
fprintf('Optimization begins ... \n');
t_w = 0;
t_c = 0;
t_r = 0;

for outerIter = 1:MaxIterEM
    
    % update T
    T = mean(W,2);
    
    % update shape and rotation
    Wc = bsxfun(@minus,W,T);
    C_pre = C;
    fval_pre = inf;
    for innerIter = 1:MaxIterAltern
        t0 = tic;
        [C,info_C] = estimateC_fused(Wc,R,dict.B,C,alpha,beta,1e-4);
        t_c = t_c + toc(t0);
        S = composeShape(dict.B,C);
        t0 = tic;
        [R,fval] = estimateR_fused(S,Wc,gamma,R,1e-4);
        t_r = t_r + toc(t0);
        fval = fval + info_C.penalty;
        if verb
            fprintf('Inner iter %d, fval = %f, [t_c,t_r] = [%.2f,%.2f] \n',...
                innerIter,fval,t_c,t_r);
        end
        if fval_pre/fval-1 < 1e-4
            break
        else
            fval_pre = fval;
        end
    end
    
    % output S is in camera frame
    S = rotateS(S,R);
    if ~isempty(S_gt)
        e = compareStructs(S_gt,S,'true');
        e = mean(e);
    else
        e = NaN;
    end
    
    if EM
        % update W by computing mean
        t0 = tic;
        for i = 1:nFrame
            W_proj = bsxfun(@plus,S(3*i-2:3*i-1,:),T(2*i-1:2*i));
            for j = 1:nJoint
                sqDist = sum(bsxfun(@minus,xy,W_proj(:,j)).^2,1)';
                likelihood = exp(-sqDist/(2*sigma^2));
                pr = (likelihood+eps) .* ...
                    reshape(heatmap(:,:,j,i)+eps,size(likelihood));
                pr = pr / sum(pr);
                W(2*i-1:2*i,j) = xy*pr;
            end
        end
        t_w = t_w + toc(t0);
    end
    
    % check convergence
    if EM
        RelChg = norm(C_pre(:)-C(:))/norm(C_pre);
        if RelChg < tol
            break
        end
        fprintf('Outer iter %d, RelChg = %f, #InnerIter = %d, error = %f \n',...
            outerIter,RelChg,innerIter,e);
    else
        break
    end
    
end

%%
W_proj = S;
W_proj(3:3:end,:) = [];
W_proj = bsxfun(@plus,W_proj,T);

output.S_init = S_init;
output.R_init = R_init;
output.C_init = C_init;
output.T_init = T_init;
output.S_final = S;
output.R_final = R;
output.C_final = C;
output.T_final = T;
output.W_init = W_init*scale;
output.W_final = W*scale;
output.W_proj = W_proj*scale;
output.size_metric = size_metric;
output.size_heatmap = size_heatmap;
output.time = t_w + t_c + t_r;
