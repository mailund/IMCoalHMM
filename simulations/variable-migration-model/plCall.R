names=c('1c1', '1c2','1c3','1c4','2c1','2c2','2c3','2c4','12m1', '12m2','12m3','12m4','21m1','21m2','21m3','21m4','rho', 'pri', 'lik','post','accept','reject','adapParam')
plCall=function(){
  par(mfrow=c(5,4))
  indexVec=(1:1000)*10
  for(i in 1:(length(tr[1,]))){
    
    if(substring(names[i],2,2)=="c" && substring(names[i],1,1)!="a"){
      plot(tr[,i], type='l', main=names[i], ylim=c(0,0.01))
      abline(h=0.002,col='red')
    }
    else if(substring(names[i],3,3)=="m"){
      plot(tr[,i], type='l', main=names[i],ylim=c(0,1000))
      abline(h=250,col="red")
    }
    else if(substring(names[i],2,2)=="h"){
      plot(tr[,i], type='l', main=names[i])
      abline(h=0.4,col="red")
    }
    else if(substring(names[i],1,2)=="ac"){
      mav <- function(x,n){filter(x,rep(1/n,n), sides=2)}
      y=mav(tr[,i],n=100)/max(tr[,i+1])
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
      plot(100:length(tr[,1]), tr[100:length(tr[,1]),i], type='l', main=names[i])
    }
    else{
      plot(tr[,i], type='l', main=names[i])
    }
  }
}