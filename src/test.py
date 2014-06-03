
from multiprocessing import Process, Queue
from IMCoalHMM.likelihood import Likelihood
from IMCoalHMM.isolation_model import IsolationModel
from pyZipHMM import Forwarder
from numpy import array

class LikelihoodThread(object):
    def __init__(self, model, input_files):
        self.model = model
        self.input_files = input_files
        self.task_queue = Queue()
        self.response_queue = Queue()
        self.log_likelihood = None

    def set_log_likelihood(self):
        forwarders = [Forwarder.fromDirectory(arg) for arg in self.input_files]
        self.log_likelihood = Likelihood(self.model, forwarders)

    def __call__(self):
        self.set_log_likelihood()

        while True:
            params = self.task_queue.get()
            logL = self.log_likelihood(params)
            self.response_queue.put(logL)




import sys
input_files = sys.argv[1:]

threads = [LikelihoodThread(IsolationModel(10), input_files) for _ in range(3)]
processes = [Process(target=thread) for thread in threads]
for process in processes:
    process.start()

threads[0].task_queue.put(array((1000, 0.001, 0.4)))
threads[1].task_queue.put(array((0.001, 2222, 0.4)))
threads[2].task_queue.put(array((1, 1, 1)))

print [thread.response_queue.get() for thread in threads]

for process in processes:
    process.terminate()
    process.join()