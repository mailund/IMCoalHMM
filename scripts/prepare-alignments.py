#!/usr/bin/env python

import os
import os.path
import sys
import gzip
from optparse import OptionParser

from pyZipHMM import Forwarder
from Bio import SeqIO


def main():
    usage = """%prog [options] <input> <input format> <output dir>

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

    parser = OptionParser(usage=usage, version="%prog 1.0")

    parser.add_option("--names",
                      dest="names",
                      type="string",
                      default=None,
                      help="A comma-separated list of names to use from the source file")
    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      action="store_true",
                      default=False,
                      help="Print status information during processing")

    options, args = parser.parse_args()

    if len(args) != 3:
        parser.error("Needs input file, input format and output file")
    in_filename = args.pop(0)
    in_format = args.pop(0)
    output_dirname = args.pop(0)

    if not os.path.exists(in_filename):
        print 'The input file', in_filename, 'does not exists.'
        sys.exit(1)

    if os.path.exists(output_dirname):
        print 'The output directory', output_dirname, 'already exists.'
        print 'If you want to replace it, please explicitly remove the current'
        print 'version first.'
        sys.exit(1)

    if in_filename.endswith('.gz'):
        if options.verbose:
            print "Assuming '%s' is a gzipped file." % in_filename
        inf = gzip.open(in_filename)
    else:
        inf = open(in_filename)

    if options.verbose:
        print "Loading data...",
        sys.stdout.flush()
    alignments = SeqIO.to_dict(SeqIO.parse(inf, in_format))
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

    os.mkdir(output_dirname)

    clean = set('ACGT')
    sequence1 = srcs[0]
    sequence2 = srcs[1]
    assert len(sequence1) == len(sequence2)
    sequence_length = len(sequence1)
    outname = os.path.join(output_dirname, 'original_sequence')
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
        print "Writing ZipHMM data to '%s'..." % output_dirname,
        sys.stdout.flush()
    f.writeToDirectory(output_dirname)
    if options.verbose:
        print "done"


if __name__ == "__main__":
    main()
