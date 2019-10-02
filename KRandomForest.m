tic
doc1 = fopen('label1.txt','r');
label = textscan(doc1,'%s');
labelDoc=label{1,1};
j=1;

feature=importdata('clusterFeature.txt');
for i =1:length(labelDoc)
   x(i,1)=labelDoc(i,:);
      
end
y=x(1:84,:);
z=y(1,1);

allX=feature(1:84,1:end);
allY=y;

RF_ensemble = TreeBagger(1,allX,allY,'Method','Classification');
    %ftsel=RF_ensemble.OOBPermutedVarDeltaError;
    error=oobError(RF_ensemble);

accuracyRF=1-error;
toc
