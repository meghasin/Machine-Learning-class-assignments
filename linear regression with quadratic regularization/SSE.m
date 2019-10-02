function[WTest]=SSE(B,lambda)

%% Variable declation with respect to test file
b=cell2mat(B);
testNormCsv=normc(b);
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

for i=1:90
    tempTest = (yTest(i)-XTest(i,:)*WtTest);
%    tempTrain = (yTrain(i)-XTrain(i,:)*WtTrain);
    for j=1:WTestLen
        tempValueTest(j) = 2*sum((-1).*tempTest*XTest(i,j))+2*lambda*abs(WtTest(j));
    end
    sse=sum((power(tempTest,2)))+lambda*power(abs(WtTest(j)),2);
    WTest = WTest-(alpha)*tempValueTest.';

end   
   % plot(tempValueTest); 