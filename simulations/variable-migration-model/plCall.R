names1=c('1c1', '1c2','1c3','1c4','2c1','2c2','2c3','2c4','12m1', '12m2','12m3','12m4','21m1','21m2','21m3','21m4','rho', 'pri', 'lik','post','accept','reject','adapParam')
names7=c('1c1', '1c2','1c3','1c4','1c5','1c6','1c7','2c1','2c2','2c3','2c4','2c5','2c6','2c7','12m1', '12m2','12m3','12m4','12m5','12m6','12m7','21m1','21m2','21m3','21m4','21m5','21m6','21m7','rho', 'pri', 'lik','post','accept','reject','adapParam')

plCall=function(names=names1,title="", epochs=4, hist=FALSE,...){
  par(mfrow=c(5,epochs))
  indexVec=(1:1000)*10
  for(i in 1:(length(tr[1,]))){
    print(names[i])
    if(substring(names[i],2,2)=="c" && substring(names[i],1,1)!="a"){
      if(!hist){
        if(paste('adapParam',names[i])%in%names){
          title2=paste(title,tr[length(tr[,1]),which(names==paste('adapParam',names[i]))])
        }
        else{
          title2=title
        }
        plot(tr[,i], type='l', main=names[i],sub=title2, ylim=c(0,0.01),...)
        abline(h=0.002,col='red')
      }
      else{
        hist(tr[,i], main=names[i], sub=title,...)
        abline(v=0.002, col='red')
      }
      
    }
    else if(substring(names[i],3,3)=="m"){
      if(!hist){
        if(paste('adapParam',names[i])%in%names){
          title2=paste(title,tr[length(tr[,1]),which(names==paste('adapParam',names[i]))])
        }
        else{
          title2=title
        }
        plot(tr[,i], type='l', main=names[i],sub=title2,ylim=c(0,2000),...)
        abline(h=500,col="red")
      }
      else{
        hist(tr[,i], main=names[i],sub=title,...)
        abline(v=500,col="red")
      }
      
    }
    else if(substring(names[i],2,2)=="h"){
      if(!hist){
        plot(tr[,i], type='l', sub=title,main=names[i],...)
        abline(h=0.4,col="red")
      }
      else{
        hist(tr[,i], main=names[i],sub=title,...)
        abline(v=0.4,col="red")
      }
      
    }
    else if(substring(names[i],1,2)=="ac"){
      mav <- function(x,n){filter(x,rep(1/n,n), sides=2)}
      y=mav(tr[,i],n=100)/max(tr[,i+1])
      print(y)
      plot(y, type='l',main=names[i],ylim=c(-0.1,1.1))
      abline(h=0.234, col='red')
      abline(h=0.05, col='red')
    }
    else if(substring(names[i],3,3)=='j' || substring(names[i],1,3)=='lik' || substring(names[i],1,2)=='pr'){
      #do nothing
      cat(".")
    }
    else if(substring(names[i],2,2)=='d'){
      plot(log(tr[,i]), type='l',sub=title, main=paste("log",names[i]))
    }
    else if(substring(names[i],1,2)=='po'){
      plot(100:length(tr[,1]), tr[100:length(tr[,1]),i], type='l',sub=title, main=names[i])
    }
    else{
      plot(tr[,i], type='l',sub=title, main=names[i])
    }
  }
}

names2=c(names1, sapply(names1[1:17], function(x){paste("adapParam",x)}))
plCall2=function(title="", hist=FALSE,...){
  plCall(names=names2,title,hist,...)
}
  
