M = dlmread('iris_train-1.csv',';');
Mtest = dlmread('iris_test-1.csv',';');
Y=M(:,end);
X=M(:,1:end-1);
X1=Mtest(:,1:end-1);
cols = {'sepal length', 'sepal width', 'petal length', 'petal width'};
col1={'sepal length'};
col2={'sepal width'};
col3={'petal length'};
col4={'petal width'};



sizeX=size(X);
valPredict={};
avgErrortot=0;
valError={};

L = 5;
errorMatrix = rand(L,1);
reqSize=sizeX(1,1);
sumErrortot=0;
i=1;
while L<31
k=10;
while k<101
for a = 0:L

colRand = randsample(cols,2);
if find(strcmp(colRand{1,1}, col1)) 
    if find(strcmp(colRand{1,2}, col2))
        Xrand=M(:,1:2);
        X=datasample(Xrand,reqSize);
        X1=datasample(Mtest(:,1:2),30);
    end
    if find(strcmp(colRand{1,2}, col3))
        Xrand=[M(:,1) M(:,3)];
        X=datasample(Xrand,reqSize);
        XrandTest=[Mtest(:,1) Mtest(:,3)];
        X1=datasample(XrandTest,30);
    end
    if find(strcmp(colRand{1,2}, col4))
        Xrand=[M(:,1) M(:,4)];
        X=datasample(Xrand,reqSize);
        XrandTest=[Mtest(:,1) Mtest(:,4)];
        X1=datasample(XrandTest,30);
    end
end
if find(strcmp(colRand{1,1}, col2)) 
    if find(strcmp(colRand{1,2}, col1))
        Xrand=M(:,1:2);
        X=datasample(Xrand,reqSize);
        X1=datasample(Mtest(:,1:2),30);
    end
    if find(strcmp(colRand{1,2}, col3))
        Xrand=M(:,2:3);
        X=datasample(Xrand,reqSize);
        X1=datasample(Mtest(:,2:3),30);
    end
    if find(strcmp(colRand{1,2}, col4))
        Xrand=[M(:,2) M(:,4)];
        X=datasample(Xrand,reqSize);
        XrandTest=[Mtest(:,2) Mtest(:,4)];
        X1=datasample(XrandTest,30);
    end
    
end
if find(strcmp(colRand{1,1}, col3)) 
    if find(strcmp(colRand{1,2}, col2))
        Xrand=M(:,2:3);
         X=datasample(Xrand,reqSize);
         X1=datasample(Mtest(:,2:3),30);
    end
    if find(strcmp(colRand{1,2}, col1))
        Xrand=[M(:,1) M(:,3)];
        X=datasample(Xrand,reqSize);
        XrandTest=[Mtest(:,1) Mtest(:,3)];
        X1=datasample(XrandTest,30);
    end
    if find(strcmp(colRand{1,2}, col4))
        Xrand=M(:,3:4);
         X=datasample(Xrand,reqSize);
         X1=datasample(Mtest(:,3:4),30);
    end
    
end
if find(strcmp(colRand{1,1}, col4)) 
    if find(strcmp(colRand{1,2}, col1))
        Xrand=([M(:,1) M(:,4)]);
         X=datasample(Xrand,reqSize);
         X1=datasample([Mtest(:,1) Mtest(:,4)],30);
    end
    if find(strcmp(colRand{1,2}, col2))
        Xrand=([M(:,2) M(:,4)]);
         X=datasample(Xrand,reqSize);
         X1=datasample([Mtest(:,2) Mtest(:,4)],30);
    end
     if find(strcmp(colRand{1,2}, col3))
        Xrand=M(:,3:4);
         X=datasample(Xrand,reqSize);
         X1=datasample(Mtest(:,3:4),30);
    end
end

t = treeBuiltUp(X,Y,colRand);

%% Display the tree
treeplot(t.p');
title('Decision tree)');
[xs,ys,h,s] = treelayout(t.p');

for i = 2:numel(t.p)
	% Get my coordinate
	my_x = xs(i);
	my_y = ys(i);

	% Get parent coordinate
	parent_x = xs(t.p(i));
	parent_y = ys(t.p(i));

	% Calculate weight coordinate (midpoint)
	mid_x = (my_x + parent_x)/2;
	mid_y = (my_y + parent_y)/2;

    % Edge label
	text(mid_x,mid_y,t.labels{i-1});
    
    % Leaf label
    if ~isempty(t.inds{i})
        val = Y(t.inds{i});
        if numel(unique(val))==1
            text(my_x, my_y, sprintf('y=%2.2f\nn=%d', val(1), numel(val)));
        else
            %inconsistent data
            text(my_x, my_y, sprintf('**y=%2.2f\nn=%d', mode(val), numel(val)));
        end
    end
end
    
%%tree = build_tree(X,Y,colRand);
tree = fitctree(X,Y,'MinLeafSize',k);
end

yp = predict(tree,X(1,:));
valPredict(i,:)={yp,L};
f_error=finderror(tree,X1,Y);
valError(i,:)={L,k,f_error};
i=i+1;
k=k+10;

end
L=L+5;

end







    
