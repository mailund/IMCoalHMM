


if __name__ == '__main__':
    from IMCoalHMM.ILS import ILSModel
    import sys
    import numpy
    from IMCoalHMM.likelihood import Likelihood
    from pyZipHMM import Forwarder

    forwarders = [Forwarder.fromDirectory(arg) for arg in sys.argv[1:]]

    log_likelihood = Likelihood(ILSModel(5, 5), forwarders)
    tau1, tau2 = 2e-6, 1e-6
    coal1, coal2, coal3, coal12, coal123 = 1000.0, 1000.0, 1000.0, 2000.0, 1000.0
    recombination_rate = 0.4
    parameters = numpy.array([tau1, tau2, coal1, coal2, coal3, coal12, coal123, recombination_rate])
    print log_likelihood(parameters)
