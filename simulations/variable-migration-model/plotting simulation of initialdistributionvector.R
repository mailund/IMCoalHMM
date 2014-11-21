setwd("~/Simresults/t30")
tr=read.table("INMmcmc-50-simAdaptTestJump-1-chain.txt",header=F, fill=TRUE)
tr=tr[-1,]
until=length(tr[,1])-5
tr2=apply(as.matrix(tr[1:until,]), c(1,2), as.numeric)
normalResults=tr2[500:until,1:24]
t11r=tr2[500:until,25:(25+20-1)]
t12r=tr2[500:until,(25+20):(25+40-1)]
t22r=tr2[500:until,(25+40):(25+60-1)]

a11=diag(var(t11r[which(normalResults[,21]==1),]))
b11=diag(var(t11r))
a11/(b11)

a12=diag(var(t12r[which(normalResults[,21]==1),]))
b12=diag(var(t12r))
a12/(b12)

a22=diag(var(t22r[which(normalResults[,21]==1),]))
b22=diag(var(t22r))
a22/(b22)

tr=normalResults
plCall()

euc

n=length(t11r[,1])
initJumps11=rep(0,n)
initJumps12=rep(0,n)
initJumps22=rep(0,n)

a=which(tr[,21]==1)
ind=a[sum(a<=500)]
bef11=tr2[ind,25:(25+20-1)]
bef12=tr2[ind,(25+20):(25+40-1)]
bef22=tr2[ind,(25+40):(25+60-1)]
for(i in 1:n){
  initJumps11[i]=dist(rbind(t11r[i,],bef11))
  initJumps12[i]=dist(rbind(t12r[i,],bef12))
  initJumps22[i]=dist(rbind(t22r[i,],bef22))
  if(normalResults[i,21]==1){
    bef11=t11r[i,]
    bef12=t12r[i,]
    bef22=t22r[i,]
  }
}

dataset=data.frame(res=normalResults[,21], sqJump=normalResults[,24], i11=initJumps11,i12=initJumps12,i22=initJumps22)
View(dataset)
g=glm(res~sqJump+i11+i12+i22, family=binomial(logit), data=dataset)
summary(g)
h=glm(res~i12, family=binomial(logit), data=dataset)
summary(h)
plot(res ~ i12, data=dataset)
lines(dataset$i12, h$fitted, type="l", col="red")
