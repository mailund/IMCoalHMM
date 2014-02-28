
from IMCoalHMM.isolation_model import IsolationModel
from pyZipHMM import Forwarder

model = IsolationModel()
pi, T, E = model.build_HMM(5, 3.0, 1.0, 4e-4)

forwarder = Forwarder.fromDirectory('examples/example_data.ziphmm')
logL = forwarder.forward(pi, T, E)
print logL