library(reshape2)
library(ggplot2)

all.chains <- rbind(data.frame(sim=1,chain=1,read.table('mcmc-sim-1-chain-1.txt', header=TRUE)[,1:3]),
                    data.frame(sim=1,chain=2,read.table('mcmc-sim-1-chain-2.txt', header=TRUE)[,1:3]),
                    data.frame(sim=1,chain=3,read.table('mcmc-sim-1-chain-3.txt', header=TRUE)[,1:3]),
                    data.frame(sim=1,chain=4,read.table('mcmc-sim-1-chain-4.txt', header=TRUE)[,1:3]),
                    data.frame(sim=1,chain=5,read.table('mcmc-sim-1-chain-5.txt', header=TRUE)[,1:3]),
                    data.frame(sim=2,chain=1,read.table('mcmc-sim-2-chain-1.txt', header=TRUE)[,1:3]),
                    data.frame(sim=2,chain=2,read.table('mcmc-sim-2-chain-2.txt', header=TRUE)[,1:3]),
                    data.frame(sim=2,chain=3,read.table('mcmc-sim-2-chain-3.txt', header=TRUE)[,1:3]),
                    data.frame(sim=2,chain=4,read.table('mcmc-sim-2-chain-4.txt', header=TRUE)[,1:3]),
                    data.frame(sim=2,chain=5,read.table('mcmc-sim-2-chain-5.txt', header=TRUE)[,1:3]))
                    

qplot(value, ..scaled.., geom='density', facets=variable~sim, color=factor(chain),
      data=melt(all.chains, id.vars=c('sim','chain'))) +
  facet_grid(sim~variable, scales='free') + theme_bw()
