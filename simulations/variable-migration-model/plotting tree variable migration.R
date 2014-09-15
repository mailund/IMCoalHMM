setwd("~/Documents/variable migration")
tr=read.table("INMmcmc-sim-1-chain.txt",header=TRUE)
names=c('1c1', '1c2','1c3','1c4','2c1','2c2','2c3','2c4','12m1', '12m2','12m3','12m4','21m1','21m2','21m3','21m4','rho', 'pri', 'lik','post','accept','reject')
par(mfrow=c(1,1))
for(i in 1:(length(tr[1,])-2)){
  plot(tr[(1:100)*100,i], type='l', main=names[i])
}

drawingOfOne=function(vectorOfParams,cmin=FALSE,mmax=FALSE, sqrtroot=5){
  
  #making the empty frame
  plot(0,0,type="n", ylim=c(0,1), xlim=c(-1.2,1.2), xaxt='n')
  #this is the number of epochs
  numberOfEpochs=floor((length(vectorOfParams)-1)/4)
  if(!cmin & !mmax){
    cmin=min(vectorOfParams[1:(numberOfEpochs*2)])
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
    widthOf1=(cmin/vectorOfParams[i])^(1/sqrtroot)
    widthOf2=(cmin/vectorOfParams[numberOfEpochs+i])^(1/sqrtroot)
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
  #...now we are here. Everyhting is plotted.
}


oopt = ani.options(interval = 0.1, nmax = 250)
mmax=max(tr[(1:250)*40,9:16])
cmin=min(tr[(1:250)*40,1:8])
## use a loop to create images one by one
for (i in 1:ani.options("nmax")) {
  drawingOfOne(tr[i*40,1:17],mmax=mmax, cmin=cmin,sqrtroot=2)
  ani.pause() ## pause for a while (
}
saveGIF({
  ani.options(nmax = 30)
  for (i in 1:ani.options("nmax")) {
    drawingOfOne(tr[i*40,1:17],mmax=mmax, cmin=cmin,sqrtroot=2)
    ani.pause() ## pause for a while (
  }
}, interval = 0.05, movie.name = "bm_demo.gif", ani.width = 600, ani.height = 600)

oopt = ani.options(interval = 0.2, nmax = 10)
## use a loop to create images one by one
for (i in 1:ani.options("nmax")) {
  drawingOfOne()
  ani.pause() ## pause for a while (
}
## restore the options
ani.options(oopt)
saveGIF({
  ani.options(nmax = 30)
  brownian.motion(pch = 21, cex = 5, col = "red", bg = "yellow")
}, interval = 0.05, movie.name = "bm_demo.gif", ani.width = 600, ani.height = 600)

ma