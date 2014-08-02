#!/usr/bin/env python

"""
A script that infers population genetic parameters using heuristic optimisers for CoalHMM.
"""

import datetime
import numpy
import os
import socket
import sys

import IMCoalHMM.genetic_algorithm
import IMCoalHMM.particle_swarm

_alignments = None
_config = None
_log_file = None
_log_to_stdout = None
_process_final_context = None
_process_logged_context = None


def _print_help():
    """
    Print the help message for the script.
    :return: None
    """
    print '''Usage: {0} configuration_file
Infers population genetic parameters using heuristic optimisers for CoalHMMs.

For an example of the mandatory configuration file, see
heuristic-optimiser.config in the directory accompanying this script.

Report bugs to Jade Cheng <ycheng@cs.au.dk>.'''.format(sys.argv[0])


def _log(text):
    """
    Log text to standard output and the log file, if specified.
    :param text: The text to log.
    :return: None
    """
    if _log_to_stdout:
        print text
    if _log_file is not None:
        with open(_log_file, 'a') as handle:
            handle.write('{0}\n'.format(text))


def _log_comment(text):
    """
    Log a "comment" line (prefixed by a '#' symbol) to the log file.
    :param text: The text to log.
    :return: None
    """
    _log('# {0}'.format(text).strip())


def _log_header():
    """
    Log the header.
    :return: None
    """
    max_lhs_len = max([len(x) for x in _config.table])

    def log_pair(lhs, rhs):
        clean_lhs = '{0}{1}'.format(lhs, ' ' * (max_lhs_len - len(lhs)))
        clean_rhs = 'None' if rhs is None else rhs
        _log_comment('{0} = {1}'.format(clean_lhs, clean_rhs))

    log_pair('time', datetime.datetime.now())
    log_pair('hostname', socket.gethostname())
    log_pair('script', os.path.basename(__file__))
    log_pair('path', os.getcwd())
    log_pair('config', sys.argv[1])
    _log_comment('')

    for key in sorted(_config.table):
        log_pair(key, _config.table[key])


class _Configuration(object):
    """
    A class that maintains settings from a configuration file consisting of lines with 'key = value' pairs.
    """

    def __init__(self, path):
        """
        Initialise a new instance of the class.
        :param path: The path to the log file.
        :return: A new instance of the class.
        """
        self.table = {}
        with open(path, 'r') as handle:
            for line_index, line in enumerate([x.strip() for x in handle]):
                if len(line) > 0 and not line.startswith('#'):
                    equals_index = line.find('=')
                    assert equals_index >= 0, 'error on line {0}: {1}'.format(line_index + 1, line)
                    key = line[:equals_index].strip()
                    value = line[equals_index + 1:].strip()
                    self.table[key] = None if 'none' == value.lower() else value

    def get_str(self, key):
        """
        Get a value from the configuration file or fail.
        :param key: The key of the setting.
        :return: The value from the configuration file.
        """
        value = self.try_str(key, None)
        assert value is not None, 'The key "{0}" is missing from the configuration.'.format(key)
        return value

    def try_float(self, key, fallback):
        """
        Get a float from the configuration file, or a specified fallback value.
        :param key: The key of the setting.
        :param fallback: The fallback value.
        :return: The value from the configuration file.
        """
        return self.__query(key, fallback, float)

    def try_int(self, key, fallback):
        """
        Get a integer from the configuration file, or a specified fallback value.
        :param key: The key of the setting.
        :param fallback: The fallback value.
        :return: The value from the configuration file.
        """
        return self.__query(key, fallback, int)

    def try_str(self, key, fallback):
        """
        Get a string from the configuration file, or a specified fallback value.
        :param key: The key of the setting.
        :param fallback: The fallback value.
        :return: The value from the configuration file.
        """
        return self.__query(key, fallback, lambda x: x)

    def try_timedelta(self, key, fallback):
        """
        Get a length of time from the configuration file, or a specified fallback value.
        :param key: The key of the setting.
        :param fallback: The fallback value.
        :return: The value from the configuration file.
        """
        return self.__query(key, fallback, lambda x: datetime.timedelta(seconds=float(x)))

    def __query(self, key, fallback, transform):
        if key not in self.table:
            self.table[key] = None if fallback is None else str(fallback)
            return fallback
        value = self.table[key]
        return None if value is None else transform(value)


class _Transformer(object):
    """
    A class that transforms a sequence of parameters by linearly interpolating each parameter between a lower and
    upper bound. The class also maintains the human-readable names of the parameters it can transform.
    """

    def __init__(self):
        """
        Initialise a new instance of the class.
        :return: A new instance of the class.
        """
        self.names = []
        self.ranges = []

    def add(self, name, lower, upper):
        """
        Add an entry for a parameter.
        :param name: The name of the parameter.
        :param lower: The lower bound for the linear interpolation.
        :param upper: The upper bound for the linear interpolation.
        :return: None
        """
        self.names.append(name)
        self.ranges.append((lower, upper))

    def transform(self, percentages):
        """
        Transform a sequence of parameters in units of percentages.
        :param percentages: The sequence of percentages.
        :return: The transformed sequence of parameters.
        """
        assert len(percentages) == len(self.names)
        transformed_sequence = []
        for index, percentage in enumerate(percentages):
            lower, upper = self.ranges[index]
            transformed_sequence.append(lower + (percentage * (upper - lower)))
        return transformed_sequence


def _execute_model_i(optimiser):
    """
    Execute an experiment for the isolation model using the specified optimiser.
    :param optimiser: The optimiser to use while executing the model.
    :return: None
    """
    from IMCoalHMM.isolation_model import IsolationModel
    from pyZipHMM import Forwarder
    from IMCoalHMM.likelihood import Likelihood

    no_states = _config.try_int('model.states', 10)

    transformer = _Transformer()
    transformer.add(
        'split_time',
        _config.try_float('model.split_time.min', 0.0),
        _config.try_float('model.split_time.max', 0.004))
    transformer.add(
        'coal_rate',
        _config.try_float('model.coal_rate.min', 0.0),
        _config.try_float('model.coal_rate.max', 2000.0))
    transformer.add(
        'recomb_rate',
        _config.try_float('model.recomb_rate.min', 0.0),
        _config.try_float('model.recomb_rate.max', 0.8))

    _log_header()

    forwarders = [Forwarder.fromDirectory(arg) for arg in _alignments]
    model = IsolationModel(no_states)
    log_likelihood = Likelihood(model, forwarders)

    def fitness_function(parameters):
        transformed_parameters = transformer.transform(parameters)
        return log_likelihood(numpy.array(transformed_parameters))

    def log_function(context):
        _process_logged_context(context, transformer)

    optimiser.log = log_function
    mle_context = optimiser.maximise(fitness_function, len(transformer.names))
    mle_parameters, mle_log_likelihood = _process_final_context(mle_context, transformer)

    mle_split_time, mle_coal_rate, mle_recomb_rate = mle_parameters
    mle_theta = 2.0 / mle_coal_rate

    _log_comment('')
    _log_comment('mle_split_time     = {0}'.format(mle_split_time))
    _log_comment('mle_coal_rate      = {0}'.format(mle_coal_rate))
    _log_comment('mle_recomb_rate    = {0}'.format(mle_recomb_rate))
    _log_comment('mle_theta          = {0}'.format(mle_theta))
    _log_comment('mle_log_likelihood = {0}'.format(mle_log_likelihood))


def _execute_model_iim(optimiser):
    """
    Execute an experiment for the isolation-with-migration model using the specified optimiser.
    :param optimiser: The optimiser to use while executing the model.
    :return: None
    """
    from pyZipHMM import Forwarder
    from IMCoalHMM.isolation_with_migration_model import IsolationMigrationModel
    from IMCoalHMM.likelihood import Likelihood

    no_ancestral_states = _config.try_int('model.ancestral_states', 10)
    no_migration_states = _config.try_int('model.migration_states', 10)

    transformer = _Transformer()
    transformer.add(
        'isolation_time',
        _config.try_float('model.isolation_time.min', 0.0),
        _config.try_float('model.isolation_time.max', 0.002))
    transformer.add(
        'mig_time',
        _config.try_float('model.mig_time.min', 0.0),
        _config.try_float('model.mig_time.max', 0.016))
    transformer.add(
        'coal_rate',
        _config.try_float('model.coal_rate.min', 0.0),
        _config.try_float('model.coal_rate.max', 2000.0))
    transformer.add(
        'recomb_rate',
        _config.try_float('model.recomb_rate.min', 0.0),
        _config.try_float('model.recomb_rate.max', 0.8))
    transformer.add(
        'mig_rate',
        _config.try_float('model.mig_rate.min', 0.0),
        _config.try_float('model.mig_rate.max', 500.0))

    _log_header()

    forwarders = [Forwarder.fromDirectory(arg) for arg in _alignments]
    model = IsolationMigrationModel(no_migration_states, no_ancestral_states)
    log_likelihood = Likelihood(model, forwarders)

    def fitness_function(parameters):
        transformed_parameters = transformer.transform(parameters)
        return log_likelihood(numpy.array(transformed_parameters))

    def log_function(context):
        _process_logged_context(context, transformer)

    optimiser.log = log_function
    mle_context = optimiser.maximise(fitness_function, len(transformer.names))
    mle_parameters, mle_log_likelihood = _process_final_context(mle_context, transformer)

    mle_isolation_time, mle_mig_time, mle_coal_rate, mle_recomb_rate, mle_mig_rate = mle_parameters
    mle_theta = 2.0 / mle_coal_rate

    _log_comment('')
    _log_comment('mle_isolation_time = {0}'.format(mle_isolation_time))
    _log_comment('mle_mig_time       = {0}'.format(mle_mig_time))
    _log_comment('mle_coal_rate      = {0}'.format(mle_coal_rate))
    _log_comment('mle_recomb_rate    = {0}'.format(mle_recomb_rate))
    _log_comment('mle_mig_rate       = {0}'.format(mle_mig_rate))
    _log_comment('mle_theta          = {0}'.format(mle_theta))
    _log_comment('mle_log_likelihood = {0}'.format(mle_log_likelihood))


def _execute_model_iim_epochs(optimiser):
    """
    Execute an experiment for the isolation-with-migration-with-epochs model using the specified optimiser.
    :param optimiser: The optimiser to use while executing the model.
    :return: None
    """
    from pyZipHMM import Forwarder
    from IMCoalHMM.isolation_with_migration_model_epochs import IsolationMigrationEpochsModel
    from IMCoalHMM.likelihood import Likelihood

    epoch_factor = _config.try_int('model.epoch_factor', 1)
    coal_count = 1 + (2 * epoch_factor)

    no_ancestral_states = _config.try_int('model.ancestral_states', 10)
    no_migration_states = _config.try_int('model.migration_states', 10)

    transformer = _Transformer()
    transformer.add(
        'isolation_time',
        _config.try_float('model.isolation_time.min', 0.0),
        _config.try_float('model.isolation_time.max', 0.002))
    transformer.add(
        'mig_time',
        _config.try_float('model.mig_time.min', 0.0),
        _config.try_float('model.mig_time.max', 0.016))
    transformer.add(
        'recomb_rate',
        _config.try_float('model.recomb_rate.min', 0.0),
        _config.try_float('model.recomb_rate.max', 0.8))
    for i in xrange(coal_count):
        key = 'coal_rate_{0}'.format(i + 1)
        transformer.add(
            key,
            _config.try_float('model.{0}.min'.format(key), 0.0),
            _config.try_float('model.{0}.max'.format(key), 2000.0))
    for i in xrange(epoch_factor):
        key = 'mig_rate_{0}'.format(i + 1)
        transformer.add(
            key,
            _config.try_float('model.{0}.min'.format(key), 0.0),
            _config.try_float('model.{0}.max'.format(key), 500.0))

    _log_header()

    forwarders = [Forwarder.fromDirectory(arg) for arg in _alignments]
    model = IsolationMigrationEpochsModel(epoch_factor, no_migration_states, no_ancestral_states)
    log_likelihood = Likelihood(model, forwarders)

    def fitness_function(parameters):
        transformed_parameters = transformer.transform(parameters)
        return log_likelihood(numpy.array(transformed_parameters))

    def log_function(context):
        _process_logged_context(context, transformer)

    optimiser.log = log_function
    mle_context = optimiser.maximise(fitness_function, len(transformer.names))
    mle_parameters, mle_log_likelihood = _process_final_context(mle_context, transformer)

    mle_isolation_time = mle_parameters[0]
    mle_mig_time = mle_parameters[1]
    mle_recomb_rate = mle_parameters[2]
    mle_coal_rates = mle_parameters[3: coal_count + 3]
    mle_mig_rates = mle_parameters[coal_count + 3:]

    _log_comment('')
    _log_comment('mle_isolation_time = {0}'.format(mle_isolation_time))
    _log_comment('mle_mig_time       = {0}'.format(mle_mig_time))
    _log_comment('mle_recomb_rate    = {0}'.format(mle_recomb_rate))
    for i, coal_rate in enumerate(mle_coal_rates):
        _log_comment('mle_coal_rate_{0}    = {1}'.format(i + 1, coal_rate))
    for i, mig_rate in enumerate(mle_mig_rates):
        _log_comment('mle_mig_rate_{0}     = {1}'.format(i + 1, mig_rate))
    _log_comment('mle_log_likelihood = {0}'.format(mle_log_likelihood))


def _find_alignments():
    """
    Find all alignment directories based on settings provided by the configuration file.
    :return: A list of alignment directories.
    """
    dirs = []
    for path in _config.get_str('alignments').split(':'):
        if os.path.exists(os.path.join(path, 'data_structure')):
            dirs.append(path)
        else:
            for entry_name in os.listdir(path):
                entry = os.path.join(path, entry_name)
                if os.path.isdir(entry):
                    dirs.append(entry)
    return dirs


def _parse_ga_crossover():
    """
    Parse the crossover settings for the genetic algorithm.
    :return: The crossover instance.
    """
    value = _config.try_str('optimiser.crossover', 'one_point')
    if value == 'one_point':
        return IMCoalHMM.genetic_algorithm.OnePointCrossover()
    if value == 'two_point':
        return IMCoalHMM.genetic_algorithm.TwoPointCrossover()
    if value == 'uniform':
        crossover = IMCoalHMM.genetic_algorithm.UniformCrossover()
        crossover.first_parent_ratio = _config.try_float(
            'optimiser.crossover.first_parent_ratio',
            crossover.first_parent_ratio)
        return crossover
    assert False, 'invalid crossover for genetic algorithm: {0}'.format(value)


def _parse_ga_initialisation():
    """
    Parse the initialisation settings for the genetic algorithm.
    :return: The initialisation instance.
    """
    value = _config.try_str('optimiser.initialisation', 'uniform')
    if value == 'fixed':
        initialisation = IMCoalHMM.genetic_algorithm.FixedInitialisation()
        initialisation.initial_value = _config.try_float(
            'optimiser.initialisation.initial_value',
            initialisation.initial_value)
        return initialisation
    if value == 'gaussian':
        initialisation = IMCoalHMM.genetic_algorithm.GaussianInitialisation()
        initialisation.mu = _config.try_float(
            'optimiser.initialisation.mu',
            initialisation.mu)
        initialisation.sigma = _config.try_float(
            'optimiser.initialisation.sigma',
            initialisation.sigma)
        return initialisation
    if value == 'uniform':
        return IMCoalHMM.genetic_algorithm.UniformInitialisation()
    assert False, 'invalid initialisation for genetic algorithm: {0}'.format(value)


def _parse_ga_mutation():
    """
    Parse the mutation settings for the genetic algorithm.
    :return: The mutation instance.
    """
    value = _config.try_str('optimiser.mutation', 'gaussian')
    if value == 'boundary':
        mutation = IMCoalHMM.genetic_algorithm.BoundaryMutation()
    elif value == 'gaussian':
        mutation = IMCoalHMM.genetic_algorithm.GaussianMutation()
        mutation.mu = _config.try_float('optimiser.mutation.mu', mutation.mu)
        mutation.sigma = _config.try_float('optimiser.mutation.sigma', mutation.sigma)
    elif value == 'uniform':
        mutation = IMCoalHMM.genetic_algorithm.UniformMutation()
    else:
        assert False, 'invalid mutation for genetic algorithm: {0}'.format(value)
    mutation.point_mutation_ratio = _config.try_float(
        'optimiser.mutation.point_mutation_ratio',
        mutation.point_mutation_ratio)
    return mutation


def _parse_ga_selection():
    """
    Parse the selection settings for the genetic algorithm.
    :return: The selection instance.
    """
    value = _config.try_str('optimiser.selection', 'tournament')
    if value == 'roulette':
        selection = IMCoalHMM.genetic_algorithm.RouletteSelection()
    elif value == 'stochastic':
        selection = IMCoalHMM.genetic_algorithm.StochasticSelection()
    elif value == 'tournament':
        selection = IMCoalHMM.genetic_algorithm.TournamentSelection()
        selection.tournament_ratio = _config.try_float(
            'optimiser.selection.tournament_ratio',
            selection.tournament_ratio)
    elif value == 'truncation':
        selection = IMCoalHMM.genetic_algorithm.TruncationSelection()
    else:
        assert False, 'invalid selection for genetic algorithm: {0}'.format(value)
    selection.selection_ratio = _config.try_float(
        'optimiser.selection.selection_ratio',
        selection.selection_ratio)
    return selection


def _parse_ga():
    """
    Parse the settings for the genetic algorithm.
    :return: The genetic algorithm instance.
    """
    optimiser = IMCoalHMM.genetic_algorithm.Optimiser()
    optimiser.elite_count = _config.try_int('optimiser.elite_count', optimiser.elite_count)
    optimiser.hall_of_fame_size = _config.try_int('optimiser.hall_of_fame_size', optimiser.hall_of_fame_size)
    optimiser.max_generations = _config.try_int('optimiser.max_generations', optimiser.max_generations)
    optimiser.population_size = _config.try_int('optimiser.population_size', optimiser.population_size)
    optimiser.timeout = _config.try_timedelta('optimiser.timeout', optimiser.timeout)
    optimiser.initialisation = _parse_ga_initialisation()
    optimiser.selection = _parse_ga_selection()
    optimiser.crossover = _parse_ga_crossover()
    optimiser.mutation = _parse_ga_mutation()
    return optimiser


def _parse_pso():
    """
    Parse the settings for the particle swarm.
    :return: The particle swarm instance.
    """
    optimiser = IMCoalHMM.particle_swarm.Optimiser()
    optimiser.max_iterations = _config.try_int('optimiser.max_iterations', optimiser.max_iterations)
    optimiser.max_initial_velocity = _config.try_float('optimiser.max_initial_velocity', optimiser.max_initial_velocity)
    optimiser.particle_count = _config.try_int('optimiser.particle_count', optimiser.particle_count)
    optimiser.omega = _config.try_float('optimiser.omega', optimiser.omega)
    optimiser.phi_particle = _config.try_float('optimiser.phi_particle', optimiser.phi_particle)
    optimiser.phi_swarm = _config.try_float('optimiser.phi_swarm', optimiser.phi_swarm)
    optimiser.timeout = _config.try_timedelta('optimiser.timeout', optimiser.timeout)
    return optimiser


def _parse_optimiser():
    """
    Parse the settings for the optimiser.
    :return: The optimiser.
    """
    global _process_final_context, _process_logged_context

    value = _config.try_str('optimiser', 'particle_swarm')
    if value == 'genetic_algorithm':
        _process_final_context = _process_final_ga_context
        _process_logged_context = _process_logged_ga_context
        return _parse_ga()
    if value == 'particle_swarm':
        _process_final_context = _process_final_pso_context
        _process_logged_context = _process_logged_pso_context
        return _parse_pso()
    assert False, 'invalid optimiser: {0}'.format(value)


def _process_final_ga_context(context, transformer):
    """
    Process the final context returned from the genetic algorithm optimiser.
    :param context: The final context returned from the genetic algorithm optimiser.
    :param transformer: The object that can transform the parameters from percentages.
    :return: A tuple of 1) the mle_parameters, and 2) the mle_log_likelihood.
    """
    from IMCoalHMM.genetic_algorithm import Context
    assert isinstance(context, Context)

    _log_comment('')
    _log_comment('iterations         = {0}'.format(context.generation))
    _log_comment('exit_condition     = {0}'.format(context.exit_condition))
    _log_comment('execution_time     = {0}'.format(context.elapsed))

    individual = context.hall_of_fame[0]
    return transformer.transform(individual.genome), individual.fitness


def _process_logged_ga_context(context, transformer):
    """
    Process a logged context returned from the genetic algorithm optimiser.
    :param context: A logged context returned from the genetic algorithm optimiser.
    :param transformer: The object that can transform the parameters from percentages.
    :return: None
    """
    from IMCoalHMM.genetic_algorithm import Context
    assert isinstance(context, Context)

    _log_comment('')
    _log_comment('best_fitness.{0} = {1}'.format(context.generation, context.hall_of_fame[0].fitness))
    _log_comment('')
    _log_comment('\t'.join(['iteration', 'individual', 'fitness'] + transformer.names))

    for i, individual in enumerate(context.population):
        x = transformer.transform(individual.genome)
        _log('\t'.join(map(str, [context.generation, i, individual.fitness] + x)))


def _process_final_pso_context(context, transformer):
    """
    Process the final context returned from the particle swarm optimiser.
    :param context: The final context returned from the particle swarm optimiser.
    :param transformer: The object that can transform the parameters from percentages.
    :return: A tuple of 1) the mle_parameters, and 2) the mle_log_likelihood.
    """
    from IMCoalHMM.particle_swarm import Context
    assert isinstance(context, Context)

    _log_comment('')
    _log_comment('iterations         = {0}'.format(context.iteration))
    _log_comment('exit_condition     = {0}'.format(context.exit_condition))
    _log_comment('execution_time     = {0}'.format(context.elapsed))

    best = context.best
    return transformer.transform(best.positions), best.fitness


def _process_logged_pso_context(context, transformer):
    """
    Process a logged context returned from the particle swarm optimiser.
    :param context: A logged context returned from the particle swarm optimiser.
    :param transformer: The object that can transform the parameters from percentages.
    :return: None
    """
    from IMCoalHMM.particle_swarm import Context
    assert isinstance(context, Context)

    _log_comment('')
    _log_comment('best_fitness.{0} = {1}'.format(context.iteration, context.best.fitness))
    _log_comment('')
    _log_comment('\t'.join(['iteration', 'particle', 'fitness'] + transformer.names))

    for i, particle in enumerate(context.particles):
        x = transformer.transform(particle.current.positions)
        _log('\t'.join(map(str, [context.iteration, i, particle.current.fitness] + x)))


def main():
    """
    Execute the main entry point of the script.
    :return: None
    """
    global _alignments, _config, _log_file, _log_to_stdout

    assert len(sys.argv) == 2, 'invalid syntax; try --help.'
    if sys.argv[1] in ('--help', '/?', '-h'):
        _print_help()
        exit()

    _config = _Configuration(sys.argv[1])

    _alignments = _find_alignments()
    _log_file = _config.try_str('log_file', None)
    _log_to_stdout = 'true' == _config.try_str('log_to_stdout', 'true').lower()

    optimiser = _parse_optimiser()

    value = _config.try_str('model', 'isolation')
    if value == 'isolation':
        _execute_model_i(optimiser)
    elif value == 'isolation_with_initial_migration':
        _execute_model_iim(optimiser)
    elif value == 'isolation_with_initial_migration_with_epochs':
        _execute_model_iim_epochs(optimiser)
    else:
        assert False, "invalid model: '{0}'.".format(value)

if __name__ == '__main__':
    main()
