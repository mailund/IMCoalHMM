library(reshape2)
library(ggplot2)

data <- read.table('constant-size-estimates.txt', header=FALSE)
thetas <- cbind(simulation=factor(1:nrow(data)), data[,-ncol(data)])

qplot(variable, value, color=simulation, data=melt(thetas, id.vars='simulation'))



