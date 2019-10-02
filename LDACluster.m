function [Result] = LDACluster()
Result=[];
c1=[];
c2=[];
B=importdata('walking.train.labels');
train=importdata('walking.train.data');
j=1;
for i=1:length(B)
   if B(i)== 0
       c1(i,:)=train(i,:);
   else
       c2(i,:)=train(i,:); 
   end
   j=j+1;
end
z=length(c1);
k=1;
for i=1:length(c1)
   if c1(i,:)~=0
       cls1(k,:)=c1(i,:);
       k=k+1
   end    
end   

l=1;
for a=1:length(c2)
   if c2(a,:)~=0
       cls2(l,:)=c2(a,:);
       l=l+1;
   end    
end   
n1=size(c1,1);
n2=size(c2,1);
m1=mean(cls1);
m2=mean(cls2);
d1=cls1-repmat(m1,size(cls1,1),1);
d2=cls2-repmat(m2,size(cls2,1),1);
% Calculate the within class variance (SW)
s1=d1'*d1;
s2=d2'*d2;
sw=s1+s2;
invsw=inv(sw);
v=invsw*(m1-m2)';
Result=v;

