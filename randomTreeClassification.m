tic
doc1 = fopen('SPAMTrainLabel.txt','r');
label = textscan(doc1,'%s');
labelDoc=label{1,1};
j=1;
feature=importdata('Features.txt');
for i =1:length(labelDoc)
    if mod( i , 2 )~=0
       x(j,1)=labelDoc(i,:);
       j=j+1;
    end
end

y=x(1:100,:);

 X=feature(1:60,1:end);
 Y=y(1:60,end);
 X1=feature(61:100,1:end);
 Y1=y(61:100,end);
% lm=size(X,1);
allX=feature(1:100,1:end);
allY=y(1:100,end);


    RF_ensemble = TreeBagger(1,allX,allY,'Method','Classification','OOBPrediction','On');
    %ftsel=RF_ensemble.OOBPermutedVarDeltaError;
    error=oobError(RF_ensemble);
    sortFt=sort(ftsel);
    reqFeature=sortFt(1,7280:7380);

accuracyRF=1-error;
toc
