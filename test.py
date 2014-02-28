
from IMCoalHMM.isolation_model import IsolationModel
from IMCoalHMM.likelihood import Likelihood
from pyZipHMM import Forwarder


model = IsolationModel()
forwarder = Forwarder.fromDirectory('examples/example_data.ziphmm')
logL = Likelihood(model, forwarder)
print logL(5, 3.0, 1.0, 4e-4)

