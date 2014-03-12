
data <- rbind(read.table('estimates.split-0.001-mya.txt',header=TRUE),
              read.table('estimates.split-0.1-mya.txt',header=TRUE),
              read.table('estimates.split-1-mya.txt',header=TRUE),
              read.table('estimates.split-2-mya.txt',header=TRUE),
              read.table('estimates.split-4-mya.txt',header=TRUE))

require(ggplot2)

qplot(factor(sim.tau), mle.tau, data=data, geom='boxplot') +
  geom_hline(yintercept=unique(data$sim.tau), col='red')

qplot(factor(sim.tau), mle.theta, data=data, geom='boxplot') +
  geom_hline(yintercept=unique(data$sim.theta), col='red')

qplot(factor(sim.tau), mle.rho, data=data, geom='boxplot') +
  geom_hline(yintercept=unique(data$sim.rho), col='red')

data <- read.table('different-optimizers.txt', header=TRUE)

qplot(optimizer, mle.tau, data=data, geom='boxplot') +
  geom_hline(yintercept=unique(data$sim.tau), col='red')

qplot(optimizer, mle.theta, data=data, geom='boxplot') +
  geom_hline(yintercept=unique(data$sim.theta), col='red')

qplot(optimizer, mle.rho, data=data, geom='boxplot') +
  geom_hline(yintercept=unique(data$sim.rho), col='red')
