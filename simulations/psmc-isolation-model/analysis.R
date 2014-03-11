library(reshape2)
library(ggplot2)
library(scales)

data <- read.table('constant-size-estimates-100Mb.txt', header=FALSE)

thetas <- cbind(simulation=factor(1:nrow(data)), data[,-ncol(data)])
names(thetas) <- c('simulation', 1:28)

qplot(as.numeric(variable), value, color=simulation, geom='line', 
      data=melt(thetas, id.vars='simulation')) +
  geom_hline(yintercept=0.001, col='red') + 
  scale_y_continuous(trans=log10_trans())#, limits=c(0.0001, 0.02))

data <- read.table('constant-size-estimates.1Gb.txt', header=FALSE)

thetas <- cbind(simulation=factor(1:nrow(data)), data[,-ncol(data)])
names(thetas) <- c('simulation', 1:28)

qplot(as.numeric(variable), value, color=simulation, geom='line', 
      data=melt(thetas, id.vars='simulation')) +
  geom_hline(yintercept=0.001, col='red') + 
  scale_y_continuous(trans=log10_trans())#, limits=c(0.0001, 0.02))
