function [Result,clust] = KmeansCluster(K,T,X);
Result=[];
B=importdata('walking.train.labels');
B=B+1;

for j=1:T
    match=0;
% k-means spike sorting
%
% K: number of classes
% X: dataset matrix (num. of cases x dimensions)


[n,d] = size(X);

% Start with random assignments
clust = ones(n,1);
for t = 1:n
    %clust(t)=0;
    %clust(t+1)=1;
    
    randomsample = randperm(K);
% Run K-means until convergence

    clust(t) = randomsample(1);
end

% Set parameters
clust2 = clust;
g = max(clust);
term = 1;

while term ~= 0;

% Find centroids
centroids = [];
%for c = 0:1
for c = 1:g;
index = find(clust == c);
if isempty(index) ~= 1;
centroids = [centroids; mean(X(index,:))];
end
end

% Find distances to centroids and recluster
[g,dim] = size(centroids);
dist = ones(n,1);
for s = 1:n;
  x = X(s,:);
x = ones(g,1)*x;
d = (centroids - x).^2;
d = sqrt(sum(d')');
[m,index] = min(d);
clust2(s) = index;
dist(s) = m;
%disp(m);

end

% Check for convergence
term = sum(clust ~= clust2);
clust = clust2;
end 
%disp(dist);
Sum=sum(dist);
Sumsquare=Sum.^2;
obj=Sumsquare/n;
disp(['Sum Squared Error for ' num2str(j) 'th iteration is:']);
disp(obj);
for i=1:n
if clust(i)==B(i)
match=match+1;
Accuracy=(match/n)*100;
end
end
disp('Correct Prediction is:');
disp(match);
disp('Accuracy is:');
disp(Accuracy);

Result=vertcat(Result,[obj, match,Accuracy]);
%disp(Result);


end