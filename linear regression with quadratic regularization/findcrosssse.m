function[c]=findcrosssse(w,a,lambda)
sumsse=0;
%% Variable declation with respect to test file
%b=cell2mat(B);

testNormCsv=normc(a);
yTest=testNormCsv(:,end);
XTest=testNormCsv(:,1:end-1);

WTest=zeros(1,45);
WTestLen = length(WTest.');
tempValueTest = WTest.';
LTest=zeros(45,1);
WtTest=WTest.';
WNorm1 = pinv((XTest')*XTest)*XTest'*yTest;
mTest=length(XTest);

alpha=0.01;

for i=1:10
    tempTest = (yTest(i)-XTest(i,:)*w.');   
    
end   
c=sum((power(tempTest,2)))+lambda*power(abs(w),2);
c=sum(c);
   