#setwd("D:/DataScience/Data_Science_Ref/R/��R���������ھ򷽷���Ӧ�ãݣ�Ѧޱ�ݣ��������ϣ�")
setwd("D:/books/dsref/��R���������ھ򷽷���Ӧ�ãݣ�Ѧޱ�ݣ��������ϣ�")
##############��ģ�����ݵ�K-Means����
set.seed(12345)
x<-matrix(rnorm(n=100,mean=0,sd=1),ncol=2,byrow=TRUE)  
x[1:25,1]<-x[1:25,1]+3           
x[1:25,2]<-x[1:25,2]-4
par(mfrow=c(2,2))
plot(x,main="�����۲��ķֲ�",xlab="",ylab="")  
#l.l.l017  
plot(x,col=(KMClu1$cluster+1),main="K-Means����K=2",xlab="",ylab="",pch=20,cex=1.5)
points(KMClu1$centers,pch=3)
set.seed(12345)
KMClu2<-kmeans(x=x,centers=4,nstart=1)   
plot(x,col=(KMClu2$cluster+1),main="K-Means����K=4,nstart=1",xlab="",ylab="",pch=20,cex=1.5)
points(KMClu2$centers,pch=3)
KMClu1$betweenss/(2-1)/KMClu1$tot.withinss/(50-2)
KMClu2$betweenss/(4-1)/KMClu2$tot.withinss/(50-4)
set.seed(12345)
KMClu2<-kmeans(x=x,centers=4,nstart=30)
plot(x,col=(KMClu2$cluster+1),main="K-Means����K=4,nstart=30",xlab="",ylab="",pch=20,cex=1.5)
points(KMClu2$centers,pch=3)

#####################K-Means����Ӧ��
PoData<-read.table(file="������Ⱦ����.txt",header=TRUE)
CluData<-PoData[,2:7]
#############K-Means����
set.seed(12345)
CluR<-kmeans(x=CluData,centers=4,iter.max=10,nstart=30)
CluR$size
CluR$centers

###########K-Means�������Ŀ��ӻ� 
par(mfrow=c(2,1))
PoData$CluR<-CluR$cluster
plot(PoData$CluR,pch=PoData$CluR,ylab="�����",xlab="ʡ��",main="��������Ա",axes=FALSE)
par(las=2)
axis(1,at=1:31,labels=PoData$province,cex.axis=0.6)
axis(2,at=1:4,labels=1:4,cex.axis=0.6)
box()
legend("topright",c("��һ��","�ڶ���","������","������"),pch=1:4,cex=0.6)
###########K-Means���������Ŀ��ӻ�
plot(CluR$centers[1,],type="l",ylim=c(0,82),xlab="�������",ylab="���ֵ(������)",main="������������ֵ�ı仯����ͼ",axes=FALSE)
axis(1,at=1:6,labels=c("������ˮ�ŷ���","������������ŷ���","�����̳��ŷ���","��ҵ��������ŷ���","��ҵ�����ŷ�����","��ҵ��ˮ�ŷ���"),cex.axis=0.6)
box()
lines(1:6,CluR$centers[2,],lty=2,col=2)
lines(1:6,CluR$centers[3,],lty=3,col=3)
lines(1:6,CluR$centers[4,],lty=4,col=4)
legend("topleft",c("��һ��","�ڶ���","������","������"),lty=1:4,col=1:4,cex=0.6)

###########K-Means����Ч���Ŀ��ӻ�����
CluR$betweenss/CluR$totss*100
par(mfrow=c(2,3))
plot(PoData[,c(2,3)],col=PoData$CluR,main="������Ⱦ���",xlab="������ˮ�ŷ���",ylab="������������ŷ���")
points(CluR$centers[,c(1,2)],col=rownames(CluR$centers),pch=8,cex=2)
plot(PoData[,c(2,4)],col=PoData$CluR,main="������Ⱦ���",xlab="������ˮ�ŷ���",ylab="�����̳��ŷ���")
points(CluR$centers[,c(1,3)],col=rownames(CluR$centers),pch=8,cex=2)
plot(PoData[,c(3,4)],col=PoData$CluR,main="������Ⱦ���",xlab="������������ŷ���",ylab="�����̳��ŷ���")
points(CluR$centers[,c(2,3)],col=rownames(CluR$centers),pch=8,cex=2)

plot(PoData[,c(5,6)],col=PoData$CluR,main="��ҵ��Ⱦ���",xlab="��ҵ��������ŷ���",ylab="��ҵ�����ŷ�����")
points(CluR$centers[,c(4,5)],col=rownames(CluR$centers),pch=8,cex=2)
plot(PoData[,c(5,7)],col=PoData$CluR,main="��ҵ��Ⱦ���",xlab="��ҵ��������ŷ���",ylab="��ҵ��ˮ�ŷ���")
points(CluR$centers[,c(4,6)],col=rownames(CluR$centers),pch=8,cex=2)
plot(PoData[,c(6,7)],col=PoData$CluR,main="��ҵ��Ⱦ���",xlab="��ҵ�����ŷ�����",ylab="��ҵ��ˮ�ŷ���")
points(CluR$centers[,c(5,6)],col=rownames(CluR$centers),pch=8,cex=2)

#################PAM����
set.seed(12345)
x<-matrix(rnorm(n=100,mean=0,sd=1),ncol=2,byrow=TRUE) 
x[1:25,1]<-x[1:25,1]+3            
x[1:25,2]<-x[1:25,2]-4
library("cluster")
set.seed(12345)
(PClu<-pam(x=x,k=2,do.swap=TRUE,stand=FALSE)) 
plot(x=PClu,data=x)

################��ξ���
PoData<-read.table(file="������Ⱦ����.txt",header=TRUE)
CluData<-PoData[,2:7]
DisMatrix<-dist(CluData,method="euclidean")
CluR<-hclust(d=DisMatrix,method="ward")

###############��ξ��������ͼ
plot(CluR,labels=PoData[,1])
box()
###########��ξ������ʯͼ
plot(CluR$height,30:1,type="b",cex=0.7,xlab="������",ylab="������Ŀ")

######ȡ4��ľ���Ⲣ���ӻ�
par(mfrow=c(2,1))
PoData$memb<-cutree(CluR,k=4)
table(PoData$memb)
plot(PoData$memb,pch=PoData$memb,ylab="�����",xlab="ʡ��",main="��������Ա",axes=FALSE)
par(las=2)
axis(1,at=1:31,labels=PoData$province,cex.axis=0.6)
axis(2,at=1:4,labels=1:4,cex.axis=0.6)
box()

#8.5#############��ϸ�˹�ֲ�ģ��
library("MASS")  
set.seed(12345)
mux1<-0    
muy1<-0    
mux2<-15    
muy2<-15    
ss1<-10   
ss2<-10    
s12<-3   
sigma<-matrix(c(ss1,s12,s12,ss2),nrow=2,ncol=2)  
Data1<-mvrnorm(n=100,mu=c(mux1,muy1),Sigma=sigma,empirical=TRUE)  
Data2<-mvrnorm(n=50,mu=c(mux2,muy2),Sigma=sigma,empirical=TRUE) 
Data<-rbind(Data1,Data2)
plot(Data,xlab="x",ylab="y")
library("mclust")
DataDens<-densityMclust(data=Data)      
plot(x=DataDens,type="persp",col=grey(level=0.8),xlab="x",ylab="y") 

#########################��ģ�����ݵ�EM����
library("mclust") 
EMfit<-Mclust(data=Data)  
summary(EMfit)
summary(EMfit,parameters=TRUE)   
plot(EMfit,"classification") 
plot(EMfit,"uncertainty")
plot(EMfit,"density")

#############����
(BIC<-mclustBIC(data=Data))
plot(BIC,G=1:7,col="black")
(BICsum<-summary(BIC,data=Data))
mclust2Dplot(Data,classification=BICsum$classification,parameters=BICsum$parameters)

###################ʵ�����ݵ�EM����
PoData<-read.table(file="������Ⱦ����.txt",header=TRUE)
CluData<-PoData[,2:7]
library("mclust") 
EMfit<-Mclust(data=CluData)  
summary(EMfit)
plot(EMfit,"BIC")
plot(EMfit,"classification")



