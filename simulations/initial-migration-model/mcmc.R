library(reshape2)
library(ggplot2)


all.chains <- rbind(data.frame(sim=1,chain=1,read.table('mcmc-sim-1-chain-1.txt', header=TRUE)[,1:5]),
                    data.frame(sim=1,chain=2,read.table('mcmc-sim-1-chain-2.txt', header=TRUE)[,1:5]),
                    data.frame(sim=1,chain=3,read.table('mcmc-sim-1-chain-3.txt', header=TRUE)[,1:5]),
                    data.frame(sim=2,chain=1,read.table('mcmc-sim-2-chain-1.txt', header=TRUE)[,1:5]),
                    data.frame(sim=2,chain=2,read.table('mcmc-sim-2-chain-2.txt', header=TRUE)[,1:5]),
                    data.frame(sim=2,chain=3,read.table('mcmc-sim-2-chain-3.txt', header=TRUE)[,1:5]))
                    
qplot(value, ..scaled.., geom='density', facets=variable~sim, color=factor(chain),
      data=melt(all.chains, id.vars=c('sim','chain'))) +
  geom_vline(aes(xintercept = true.value), 
             data=data.frame(variable=c('isolation.period','migration.period','theta','rho','migration'),
                             true.value=c(0.001, 0.003, 0.002, 0.4, 250)),
             col=I('black'), linetype=I('dashed')) +
  facet_grid(sim~variable, scales='free') + theme_bw()
