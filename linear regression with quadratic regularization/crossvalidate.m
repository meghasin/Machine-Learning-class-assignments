function [w,A] =crossvalidate(lambda)
m=csvread('trainp1.csv');
w=cell(10,1);
%A=cell(10,1);
trainnorm=normc(m);
K = mat2cell(trainnorm, repmat(10, 10, 1), 46);
for i=1:10
    A(i)=K(i);
    B=vertcat(K(1:i-1,1:end),K(i+1:end,1:end));
    w{i}= SSE(B,lambda);
    %ssek(i)=SSE(A(i),lambda);
end


  