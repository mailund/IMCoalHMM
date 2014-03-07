#!/usr/bin/env python

import os
import os.path
import sys
import gzip
from argparse import ArgumentParser

from pyZipHMM import Forwarder
from Bio import SeqIO


def main():
    usage = """%(prog)s [options] <input> <input format> <output dir>

This program reads in an input sequence in any format supported by BioPython
and writes out a preprocessed file ready for use with zipHMM.
Also supports gzipped input files, if the name ends with `.gz`.

Assumption #1: Either the file is a pairwise alignment, or you have provided
exactly two names to the `--names` option.

Assumption #2: The file uses a simple ACGT format (and N/-). Anything else will
be interpreted as N and a warning will be given with all unknown symbols.

Warning: This program uses SeqIO.to_dict to read in the entire alignment, you
may want to split the alignment first if it's very large.
"""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")

    parser.add_argument("--names",
                        type=str,
                        default=None,
                        help="A comma-separated list of names to use from the source file")
    parser.add_argument("--verbose",
                        action="store_true",
                        default=False,
                        help="Print status information during processing")

    # positional arguments
    parser.add_argument("in_filename", type=str, help="Input file")
    parser.add_argument("in_format", type=str, help="The file format for the input")
    parser.add_argument("output_dirname", type=str, help="Where to write the ZipHMM alignment")

    options = parser.parse_args()

    if not os.path.exists(options.in_filename):
        print 'The input file', options.in_filename, 'does not exists.'
        sys.exit(1)

    if os.path.exists(options.output_dirname):
        print 'The output directory', options.output_dirname, 'already exists.'
        print 'If you want to replace it, please explicitly remove the current'
        print 'version first.'
        sys.exit(1)

    if options.in_filename.endswith('.gz'):
        if options.verbose:
            print "Assuming '%s' is a gzipped file." % options.in_filename
        inf = gzip.open(options.in_filename)
    else:
        inf = open(options.in_filename)

    if options.verbose:
        print "Loading data...",
        sys.stdout.flush()
    alignments = SeqIO.to_dict(SeqIO.parse(inf, options.in_format))
    if options.verbose:
        print "done"

    if options.names:
        names = options.names.split(',')
    else:
        names = list(alignments.keys())

    if len(names) != 2:
        print 'There must be exactly two species names specified!'
        sys.exit(1)

    if options.verbose:
        print "Assuming pairwise alignment between '%s' and '%s'" % (names[0], names[1])
    srcs = [alignments[name].seq for name in names]

    os.mkdir(options.output_dirname)

    clean = set('ACGT')
    sequence1 = srcs[0]
    sequence2 = srcs[1]
    assert len(sequence1) == len(sequence2)
    sequence_length = len(sequence1)
    outname = os.path.join(options.output_dirname, 'original_sequence')
    if options.verbose:
        print "Writing file readable by ZipHMM to '%s'..." % outname,
        sys.stdout.flush()
    seen = set()
    with open(outname, 'w', 64 * 1024) as f:
        for i in xrange(sequence_length):
            s1, s2 = sequence1[i].upper(), sequence2[i].upper()
            seen.add(s1)
            seen.add(s2)
            if s1 not in clean or s2 not in clean:
                print >> f, 2,
            elif s1 == s2:
                print >> f, 0,
            else:
                print >> f, 1,
    if options.verbose:
        print "done"
    if len(seen - set('ACGTN-')) > 1:
        print >> sys.stderr, "I didn't understand the following symbols form the input sequence: %s" % (
            ''.join(list(seen - set('ACGTN-'))))
    if options.verbose:
        print "ZipHMM is preprocessing...",
        sys.stdout.flush()
    f = Forwarder.fromSequence(seqFilename=outname,
                               alphabetSize=3, minNoEvals=500)
    if options.verbose:
        print "done"

    if options.verbose:
        print "Writing ZipHMM data to '%s'..." % options.output_dirname,
        sys.stdout.flush()
    f.writeToDirectory(options.output_dirname)
    if options.verbose:
        print "done"


if __name__ == "__main__":
    main()
