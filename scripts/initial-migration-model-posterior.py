#!/usr/bin/env python

"""Script for posterior decoding in an isolation model.
"""

from argparse import ArgumentParser

from IMCoalHMM.isolation_with_migration_model import IsolationMigrationModel
from pyZipHMM import posteriorDecoding

import os


# Hack to avoid error messages when pipe-lining through head
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)


def write_posterior_table(fout, seqname, pd_table):
    no_states = pd_table.getHeight()
    seq_len = pd_table.getWidth()

    pos = 1
    for line in xrange(seq_len):
        fout.write('%s\t%d' % (seqname, pos))
        pos += 1
        for state in xrange(no_states):
            fout.write('\t%f' % pd_table[state, line])
        fout.write('\n')

    fout.flush()


def main():
    """
    Run the main script.
    """
    usage = """%(prog)s [options] <forwarder dirs>

This program outputs the posterior decoding for each position in an alignment
from the initial migration model."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument("--ancestral-states",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time in the ancestral population (10)")
    parser.add_argument("--migration-states",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time in the migration period (10)")

    parser.add_argument("--isolation-period", type=float, help="Length of the isolation period in substitutions")
    parser.add_argument("--migration-period", type=float, help="Length of the migration period in substitutions")
    parser.add_argument("--theta", type=float, help="Effective population size in 4Ne substitutions")
    parser.add_argument("--rho",   type=float, help="Recombination rate in substitutions")
    parser.add_argument("--migration-rate", type=float, help="Migration rate in number of migrations per substitution")

    parser.add_argument('alignments', nargs='+', help='Alignments in ZipHMM format')

    options = parser.parse_args()
    if len(options.alignments) < 1:
        parser.error("Input alignment not provided!")

    if options.isolation_period is None or options.migration_period is None or \
                    options.theta is None or options.rho is None or options.migration_rate is None:
        parser.error("All model parameters must be provided for posterior decoding.")

    # get options
    no_migration_states = options.migration_states
    no_ancestral_states = options.ancestral_states
    isolation = options.isolation_period
    migration = options.migration_period
    coal = 1 / (options.theta / 2)
    rho = options.rho
    mig_rate = options.migration_rate
    parameters = (isolation, migration, coal, rho, mig_rate)

    model = IsolationMigrationModel(no_migration_states, no_ancestral_states)
    emission_points = model.emission_points(isolation, migration, coal, rho, mig_rate)
    init_probs, trans_probs, emission_probs = model.build_hidden_markov_model(parameters)

    try:
        with open(options.outfile, 'w') as outfile:
            print >> outfile, '## Model: Initial migration'
            print >> outfile, '## Parameters: isolation-period={0} migration-period={1}'.format(isolation, migration),
            print >> outfile, 'theta={0} rho={1} migration-rate={2}'.format(options.theta, rho, mig_rate)
            print >> outfile, '## Emission points:', ' '.join(map(str, emission_points))

            for arg in options.alignments:
                print >> outfile, '## Sequence:', arg
                seqfile = os.path.join(arg, 'original_sequence')
                _, pd_table = posteriorDecoding(seqfile, init_probs, trans_probs, emission_probs)
                write_posterior_table(outfile, arg, pd_table)
                outfile.flush()

    except IOError: # To avoid broken pipes when used in a pipeline
        pass

if __name__ == '__main__':
    main()
