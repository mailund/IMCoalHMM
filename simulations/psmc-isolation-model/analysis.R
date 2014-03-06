
data <- read.table('constant-size-estimates.txt', header=FALSE)
thetas <- data[,-ncol(data)]


plot(1:ncol(thetas), thetas[1,])
