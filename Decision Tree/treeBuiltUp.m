function t = treeBuiltUp(X,Y,cols)


% Create an empty decision tree, which has one node and everything in it
inds = {1:size(X,1)}; % A cell per node containing indices of all data in that node
p = 0; % Vector contiaining the index of the parent node for each node
labels = {}; % A label for each node

% Create tree by splitting on the root
z=[inds p labels] ;
[inds p labels] = split_node(X, Y, inds, p,labels, cols, 1);


t.inds = inds;
t.p = p;
t.labels = labels;



function [inds p labels] = split_node(X, Y, inds, p, labels, cols , node)
% Recursively splits nodes based on information gain

sample_count=size(X(inds{node}),2);
if (sample_count<=50)
   return;
end

best_ig = -inf; %best information gain
best_feature = 0; %best feature to split on
best_val = 0; % best value to split the best feature on

curr_X = X(inds{node},:);
curr_Y = Y(inds{node});
% Loop over each feature
for i = 1:size(X,2)
    feat = curr_X(:,i);
    
    % Deterimine the values to split on
    vals = unique(feat);
    splits = 0.5*(vals(1:end-1) + vals(2:end));
    if numel(vals) < 2
        continue
    end
    
    % Get binary values for each split value
    bin_mat = double(repmat(feat, [1 numel(splits)]) < repmat(splits', [numel(feat) 1]));
    
    % Compute the information gains
    H = ent(curr_Y);
    H_cond = zeros(1, size(bin_mat,2));
    for j = 1:size(bin_mat,2)
        H_cond(j) = cond_ent(curr_Y, bin_mat(:,j));
    end
    IG = H - H_cond;
    
    % Find the best split
    [val ind] = max(IG);
    if val > best_ig
        best_ig = val;
        best_feature = i;
        best_val = splits(ind);
    end
end

% Split the current node into two nodes
feat = curr_X(:,best_feature);
feat = feat < best_val;
inds = [inds; inds{node}(feat); inds{node}(~feat)];
inds{node} = [];
p = [p; node; node];
labels = [labels; sprintf('%s < %2.2f', cols{best_feature}, best_val); ...
    sprintf('%s >= %2.2f', cols{best_feature}, best_val)];

% Recurse on newly-create nodes
n = numel(p)-2;
[inds p labels] = split_node(X, Y, inds, p, labels, cols, n+1);
[inds p labels] = split_node(X, Y, inds, p, labels, cols, n+2);


