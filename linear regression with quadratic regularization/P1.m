
testCsv=csvread('test p1-16.csv');
trainCsv=csvread('train p1-16.csv');
%% Variable declation with respect to test file
testNormCsv=normc(testCsv);
yTest=testNormCsv(:,end);
XTest=testNormCsv(:,1:end-1);
WTest=zeros(1,45);
WTestLen = length(WTest.');
tempValueTest = WTest.';
LTest=zeros(45,1);
WtTest=WTest.';
WNorm1 = pinv((XTest')*XTest)*XTest'*yTest;
mTest=length(XTest);

%% Variable declaration with respect to train file
trainNormCsv=normc(trainCsv);
yTrain=trainNormCsv(:,end);
XTrain=trainNormCsv(:,1:end-1);
WTrain=zeros(1,45);
WTrainLen = length(WTrain.');
tempValueTrain = WTrain.';
LTrain=zeros(45,1);
WtTrain=WTrain.';
WNormTrain1 = pinv((XTrain')*XTrain)*XTrain'*yTrain;


mTrain=length(XTrain);
alpha=0.9;
disp('MATLAB---');
while alpha > 0.1
    lambda=0.001;
    while lambda<101
        for i=1:100
            tempTest = (yTest(i)-XTest(i,:)*WtTest);
            tempTrain = (yTrain(i)-XTrain(i,:)*WtTrain);
            for j=1:WTestLen
                tempValueTest(j) = sum(tempTest*(-1)*XTest(i,j))+2*lambda*abs(WtTest(j));
            end
            for j=1:WTrainLen
                tempValueTrain(j) = sum(tempTrain*(-1)*XTrain(i,j))+2*lambda*abs(WtTrain(j));
            end
        %WTest = WTest - (alpha)*tempValueTest;
        %WNormTest=sqrt(power(WTest(i),2));
            WTrain = WTrain - (alpha)*tempValueTrain;
            WNormTrain=sqrt(power(WTrain(i),2));
            aTest=(power(tempTest,2));
            aTrain=(power(tempTrain,2));
            LTest = sum(aTest);
            LTrain =sum(aTrain);
            LRTest = LTest+lambda*power(WNormTrain,2);
            %disp(tempValueTest);
            %plot(tempValueTest);    
            LRTrain = LTrain+lambda*power(WNormTrain,2);
            %surf(LRTrain);
            T=table(alpha,lambda,LRTest,LRTrain, LTrain);
        end
        if lambda==0
            lambda=1000;
        else
            if lambda==100
                lambda=0;
            else
                lambda=lambda*10;
            end
        end    
        disp(T);
    end
    alpha=alpha-0.1;
    disp('alpha iter');
    disp(T);
end
%disp(T);
 
  