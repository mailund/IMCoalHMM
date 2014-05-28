#!/usr/bin/env python

"""Script for posterior decoding in an isolation model.
"""

from argparse import ArgumentParser

from IMCoalHMM.isolation_model import IsolationModel
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
from the isolation model."""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument("--states",
                        type=int,
                        default=10,
                        help="Number of intervals used to discretize the time (10)")

    parser.add_argument("--split", type=float, help="Split time in substitutions")
    parser.add_argument("--theta", type=float, help="Effective population size in 4Ne substitutions")
    parser.add_argument("--rho",   type=float, help="Recombination rate in substitutions")

    parser.add_argument('alignments', nargs='+', help='Alignments in ZipHMM format')

    options = parser.parse_args()
    if len(options.alignments) < 1:
        parser.error("Input alignment not provided!")

    if options.split is None or options.theta is None or options.rho is None:
        parser.error("All model parameters must be provided for posterior decoding.")

    # get options
    no_states = options.states
    split = options.split
    coal = 1 / (options.theta / 2)
    rho = options.rho

    model = IsolationModel(no_states)
    emission_points = model.emission_points(split, coal, None)
    init_probs, trans_probs, emission_probs = model.build_hidden_markov_model((split, coal, rho))

    try:
        with open(options.outfile, 'w') as outfile:
            print >> outfile, '## Model: Isolation'
            print >> outfile, '## Parameters: split={0} theta={1} rho={2}'.format(split, options.theta, rho)
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
