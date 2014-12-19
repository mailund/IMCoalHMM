setwd("~/Documents/variable migration")
tr=read.table("INMmcmc-sim-1-chain.txt",header=TRUE)
names=c('1c1', '1c2','1c3','1c4','2c1','2c2','2c3','2c4','12m1', '12m2','12m3','12m4','21m1','21m2','21m3','21m4','rho', 'pri', 'lik','post','accept','reject','adapParam')
plCall=function(){
par(mfrow=c(5,4))
indexVec=(1:1000)*10
for(i in 1:(length(tr[1,]))){
  
  if(substring(names[i],2,2)=="c" && substring(names[i],1,1)!="a"){
    plot(tr[indexVec,i], type='l', main=names[i], ylim=c(0,0.01))
    abline(h=0.002,col='red')
  }
  else if(substring(names[i],3,3)=="m"){
    plot(tr[indexVec,i], type='l', main=names[i],ylim=c(0,1000))
    abline(h=250,col="red")
  }
  else if(substring(names[i],2,2)=="h"){
    plot(tr[indexVec,i], type='l', main=names[i])
    abline(h=0.4,col="red")
  }
  else if(substring(names[i],1,2)=="ac"){
    mav <- function(x,n){filter(x,rep(1/n,n), sides=2)}
    y=mav(tr[,i],n=10)/max(tr[,i+1])
    plot(y, type='l',main=names[i],ylim=c(0,0.5))
    abline(h=0.234, col='red')
    abline(h=0.05, col='red')
  }
  else if(substring(names[i],3,3)=='j' || substring(names[i],1,3)=='lik' || substring(names[i],1,2)=='pr'){
    #do nothing
    cat(".")
  }
  else if(substring(names[i],2,2)=='d'){
    plot(log(tr[,i]), type='l', main=paste("log",names[i]))
  }
  else if(substring(names[i],1,2)=='po'){
    plot(100:10000, tr[100:length(tr[,1]),i], type='l', main=names[i])
  }
  else{
    plot(tr[,i], type='l', main=names[i])
  }
}
plot(tr[,length(tr[1,])], type='l')
plot(tr[(1:1000)*10,1],type='l', ylim=c(0,1))
plot(cumsum(tr)

par(mfrow=c(1,1))
drawingOfOne=function(vectorOfParams,cmax=FALSE,mmax=FALSE, sqrtroot=5){
  
  #making the empty frame
  plot(0,0,type="n", ylim=c(0,1), xlim=c(-1.2,1.2), xaxt='n')
  #this is the number of epochs
  numberOfEpochs=floor((length(vectorOfParams)-1)/4)
  if(!cmax & !mmax){
    cmax=max(vectorOfParams[1:(numberOfEpochs*2)])
    mmax=max(vectorOfParams[(numberOfEpochs*2+1):(numberOfEpochs*4)])
  }
  x1leftRemember=-0.6
  x1rightRemember=-0.6
  x2leftRemember=0.6
  x2rightRemember=0.6
  #I make the tree starting from the bottom(i==1)...
  for(i in 1:numberOfEpochs){
    sizeOf12=(vectorOfParams[numberOfEpochs*2+i]/mmax)^(1/sqrtroot)/numberOfEpochs*4/11
    sizeOf21=(vectorOfParams[numberOfEpochs*3+i]/mmax)^(1/sqrtroot)/numberOfEpochs*4/11
    widthOf1=(vectorOfParams[i]/cmax)^(1/sqrtroot)
    widthOf2=(vectorOfParams[numberOfEpochs+i]/cmax)^(1/sqrtroot)
    x1left=-0.6-widthOf1/2
    x1right=-0.6+widthOf1/2
    x2left=0.6-widthOf2/2
    x2right=0.6+widthOf2/2
    yfrom=(i-1)/numberOfEpochs
    yto=i/numberOfEpochs
    #the migration arrows contain each a maximum of 4/11 of the whole
    y1arrowHeight=yfrom+3/11/numberOfEpochs
    y2arrowHeight=yfrom+8/11/numberOfEpochs
    y1lower=y1arrowHeight-sizeOf12/2
    y1upper=y1arrowHeight+sizeOf12/2
    y2lower=y2arrowHeight-sizeOf21/2
    y2upper=y2arrowHeight+sizeOf21/2
    lines(c(x1left,x1left),c(yfrom,yto))
    lines(c(x1right,x1right), c(yfrom,y1lower))
    lines(c(x1right,x1right), c(y1upper,y2lower))
    lines(c(x1right,x1right), c(y2upper,yto))
    lines(c(x1right,x2left), c(y1lower,y1lower))
    lines(c(x1right,x2left), c(y2lower,y2lower))
    lines(c(x1right,x2left), c(y1upper,y1upper))
    lines(c(x1right,x2left), c(y2upper,y2upper))
    lines(c(x2right,x2right), c(yfrom,yto))
    lines(c(x2left,x2left), c(yfrom,y1lower))
    lines(c(x2left,x2left), c(y1upper,y2lower))
    lines(c(x2left,x2left), c(y2upper,yto))
    lines(c(x1leftRemember,x1left), c(yfrom,yfrom))
    lines(c(x2leftRemember,x2left), c(yfrom,yfrom))
    lines(c(x1rightRemember,x1right), c(yfrom,yfrom))
    lines(c(x2rightRemember,x2right), c(yfrom,yfrom))
    arrows(x0=x1right, y0=y1arrowHeight, x1=x2left, col='red')
    arrows(x0=x2left, y0=y2arrowHeight, x1=x1right, col='blue')
    x1leftRemember=x1left
    x2leftRemember=x2left
    x1rightRemember=x1right
    x2rightRemember=x2right
  }
  text(x=-1,y=1, labels=c(round(vectorOfParams[numberOfEpochs*4+1],digits=4)))
  #...now we are here. Everyhting is plotted.
}

par(mfrow=c(1,1))
library(animation)
oopt = ani.options(interval = 0.1, nmax = 250)
mmax=max(tr[(1:250)*40,9:16])
cmax=max(tr[(1:250)*40,1:8])
## use a loop to create images one by one
for (i in 1:ani.options("nmax")) {
  drawingOfOne(tr[i*40,1:17],mmax=mmax, cmax=cmax,sqrtroot=1)
  ani.pause() ## pause for a while (
}

setwd("~/Simresults/t9")
tr=read.table("INMmcmc-smallVar-sim-1-chain.txt",header=TRUE, fill=TRUE)
tr=read.table("INMmcmc-QuitesmallVar-sim-1-chain.txt",header=TRUE, fill=TRUE)
tr=read.table("INMmcmc-VerysmallVar-sim-1-chain.txt",header=TRUE, fill=TRUE)
tr=tr[1:10000,]
tr=apply(tr,c(1,2),as.numeric)
exp(-1)
rMeans=function(x,blocks){
  
}
mean(tr[,length(tr[1,])])
t=tr[,length(tr[1,])]

cov(log(tr))

setwd("~/Simresults/t19")
tr=read.table("INMmcmc-.1-simAdapt-1-chain.txt",header=TRUE, fill=TRUE)
tr=tr[1:(length(tr[,1])-3),]
tr2=as.matrix(tr)
tr=tr2[which(as.vector(sapply(tr2[,19],as.numeric))<0),]
tr=apply(tr, c(1,2), as.numeric)
tr=tr[,-1]
typeof(tr[,20])
typeof(c(2,3,1,1,1))
print(tr[,20])
plot(jitter(sapply(tr[,21],as.numeric)/100), pch=22, cex=0.3)
mean(sapply(tr[,21],as.numeric)/100)

#anden strategi
setwd("~/Simresults/t28")
tr=read.table("INMmcmc-50d-simAdapt-1-chain.txt",header=TRUE, fill=TRUE)
tr=read.table("INMmcmc-1-simTreesMM-2-chain.txt",header=TRUE, fill=TRUE)
tr=tr[1:(length(tr[,1])-3),-length(tr[1,])]
tr=tr[1:(length(tr[,1])-3),]
tr2=as.matrix(tr)
tr=tr2[which(as.vector(tr2[,length(tr[1,])]!="")),]
tr=tr[1:10000,]
tr=apply(tr, c(1,2), as.numeric)
#der kan komme na'er men det gør ikke noget for det er i den sidste søjle
plCall()

n=1:10000
plot(10*cumsum(1/sqrt(n)))

setwd("~")
tr=read.table("INMmcmc-smallVar-sim-1-chain.txt",header=FALSE, fill=TRUE, skip=1)
tr=
tr=tr[1:(length(tr[,1])-4),]
tr2=as.matrix(tr)
tr=tr2[which(as.vector(tr2[,length(tr[1,])]!="")),]
tr=apply(tr, c(1,2), as.numeric)

