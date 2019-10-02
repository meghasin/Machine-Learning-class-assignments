function[error] = finderror(T,X,Y)
match=0;
error=0;
iter=size(X,1);
%sizeX=size(X);
%reqSize=sizeX(1,1);
for i=1:iter
      pre(i)=predict(T,X(i,:));
end

for i=1:iter
    if pre(i)==Y(i)
    match=match+1;
    end
end
error=iter-match;
error=error/iter;
error=error*100;