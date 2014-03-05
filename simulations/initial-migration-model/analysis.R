
data <- rbind(read.table('estimates.split-1mya.mig-0.5mya.mig-0.3.txt',header=TRUE),
              read.table('estimates.split-2mya.mig-0.5mya.mig-0.1.txt',header=TRUE),
              read.table('estimates.split-2mya.mig-0.5mya.mig-0.3.txt',header=TRUE),
              read.table('estimates.split-2mya.mig-0.5mya.mig-0.5.txt',header=TRUE),
              read.table('estimates.split-2mya.mig-0.5mya.mig-1.0.txt',header=TRUE),
              read.table('estimates.split-2mya.mig-0.5mya.mig-2.0.txt',header=TRUE),
              read.table('estimates.split-4mya.mig-0.5mya.mig-0.1.txt',header=TRUE),
              read.table('estimates.split-4mya.mig-0.5mya.mig-0.3.txt',header=TRUE),
              read.table('estimates.split-4mya.mig-0.5mya.mig-0.5.txt',header=TRUE),
              read.table('estimates.split-4mya.mig-0.5mya.mig-1.0.txt',header=TRUE),
              read.table('estimates.split-4mya.mig-0.5mya.mig-2.0.txt',header=TRUE),
              read.table('estimates.split-6mya.mig-0.5mya.mig-0.1.txt',header=TRUE),
              read.table('estimates.split-6mya.mig-0.5mya.mig-0.3.txt',header=TRUE),
              read.table('estimates.split-6mya.mig-0.5mya.mig-0.5.txt',header=TRUE),
              read.table('estimates.split-6mya.mig-0.5mya.mig-1.0.txt',header=TRUE),
              read.table('estimates.split-6mya.mig-0.5mya.mig-2.0.txt',header=TRUE))

library(ggplot2)
library(scales)

qplot(factor(sim.mig), mle.isolation.period, data=data, geom='boxplot',
      fill=factor(sim.migration.period)) +
  geom_hline(yintercept=unique(data$sim.isolation.period), col='red')

qplot(factor(sim.migration.period), mle.isolation.period, data=data, geom='boxplot',
      fill=factor(sim.migration.period)) +
  geom_hline(yintercept=unique(data$sim.isolation.period), col='red')



qplot(factor(sim.mig), mle.migration.period, data=data, geom='boxplot', 
      fill=factor(sim.migration.period)) +
  geom_hline(yintercept=unique(data$sim.migration.period), col='red')

qplot(factor(sim.migration.period), mle.migration.period, data=data, geom='boxplot',
      fill=factor(sim.mig)) +
  geom_hline(yintercept=unique(data$sim.migration.period), col='red')


qplot(factor(sim.mig), mle.theta, data=data, geom='boxplot',
      fill=factor(sim.migration.period)) +
  geom_hline(yintercept=unique(data$sim.theta), col='red')

qplot(factor(sim.mig), mle.mig, data=data, geom='boxplot', fill=factor(sim.migration.period)) +
  geom_hline(yintercept=unique(data$sim.mig), col='red') +
  scale_y_continuous(trans=log10_trans())
