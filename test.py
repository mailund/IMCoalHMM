
from IMCoalHMM.isolation_model import IsolationModel
from IMCoalHMM.likelihood import Likelihood
from pyZipHMM import Forwarder

model = IsolationModel()
forwarder = Forwarder.fromDirectory('examples/example_data.ziphmm')
logL = Likelihood(model, forwarder)

no_states = 10
Ne = 20000
u = 1e-9
g = 25

split_time = 5e6 * u            # 5 mya in substitutions
coal_rate = 1.0/(2*Ne*u*g)      # 1/(theta/2)
recomb_rate = 0.01/1e6 / (g*u)  # 1 cM/Mb in substitutions


from scipy import linspace
from pylab import plot, show

# log-likelihood of split times
split_times = linspace(0.5*split_time, 1.5*split_time)
logLs = [logL(no_states, t, coal_rate, recomb_rate) for t in split_times]
plot(split_times, logLs)
show()

# log-likelihood of coalescence rates
coal_rates = linspace(0.5*coal_rate, 10*coal_rate)
logLs = [logL(no_states, split_time, cr, recomb_rate) for cr in coal_rates]
plot(coal_rates, logLs)
show()

# log-likelihood of recombination rates
recomb_rates = linspace(0.1*recomb_rate, 1.0*recomb_rate)
logLs = [logL(no_states, split_time, coal_rate, rr) for rr in recomb_rates]
plot(recomb_rates, logLs)
show()
