#!/usr/bin/env python

import os
import os.path
import sys
import gzip
from argparse import ArgumentParser
from random import randint

from pyZipHMM import Forwarder
from Bio import SeqIO

def randomConverter(x):
    if x=='R':
        if randint(0,1)==0:
            return 'A'
        else:
            return 'G'
    if x=='Y':
        if randint(0,1)==0:
            return 'T'
        else:
            return 'C'
    if x=='K':
        if randint(0,1)==0:
            return 'G'
        else:
            return 'T'
    if x=='M':
        if randint(0,1)==0:
            return 'A'
        else:
            return 'C'
    if x=='S':
        if randint(0,1)==0:
            return 'C'
        else:
            return 'G'
    if x=='W':
        if randint(0,1)==0:
            return 'A'
        else:
            return 'T'
    if x=='B':
        r=randint(0,2)
        if r==0:
            return 'C'
        if r==1:
            return 'T'
        return 'G'
    if x=='D':
        r=randint(0,2)
        if r==0:
            return 'A'
        if r==1:
            return 'T'
        return 'G'
    if x=='H':
        r=randint(0,2)
        if r==0:
            return 'C'
        if r==1:
            return 'T'
        return 'A'
    if x=='V':
        r=randint(0,2)
        if r==0:
            return 'C'
        if r==1:
            return 'G'
        return 'A'
    return x

def main():
    usage = """%(prog)s [options] <input> <input format> <output dir>

This program reads in an input sequence in any format supported by BioPython
and writes out a preprocessed file ready for use with zipHMM.
Also supports gzipped input files, if the name ends with `.gz`.

Assumption #1: Either the file is a pairwise alignment, or you have provided
exactly two names to the `--names` option.

Assumption #2: The file uses a simple ACGT format (and N/-). Anything else will
be interpreted aetup)==options.parallels:
                print "The requested number of parallel chains do not match those requested"
        if not options.mc3_mcg_setup:
            chain_structure=[1]*options.parallels
        else:
            chain_structure=options.mc3_mcg_setup
        adapts=[] #we have to make a list of adaptors
        no_chains=len(chain_structure)s N and a warning will be given with all unknown symbols.

Warning: This program uses SeqIO.to_dict to read in the entire alignment, you
may want to split the alignment first if it's very large.
"""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.1")

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
    parser.add_argument("--where_path_ends",type=int, default=0, help="The path is assumed to consist of a lot of /, \
        and it is assumed that the path goes through the working directory. This is the index[counting 0] of the working directory")

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

    clean = set('ACGT')

    if options.names:
        names = options.names.split(',')
    else:
        names = list(alignments.keys())


    if len(names) == 2:
        # PAIRWISE ALIGNMENT ###########################################################################
        if options.verbose:
            print "Assuming pairwise alignment between '%s' and '%s'" % (names[0], names[1])
        srcs = [alignments[name].seq for name in names]

        os.mkdir(options.output_dirname)
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

                
                if s1 not in set('ACGTN-'):
                    s1=randomConverter(s1)
                if s2 not in set('ACGTN-'):
                    s2=randomConverter(s2)
                    
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
            print "ZipHMM is pre-processing...",
            sys.stdout.flush()
        f = Forwarder.fromSequence(seqFilename=outname, alphabetSize=3, minNoEvals=500)
        if options.verbose:
            print "done"

    elif len(names) == 3:
        # TRIPLET ALIGNMENT ###########################################################################
        if options.verbose:
            print "Assuming triplet alignment between '%s', '%s', and '%s'" % (names[0], names[1], names[2])
        srcs = [alignments[name].seq for name in names]

        os.mkdir(options.output_dirname)
        sequence1 = srcs[0]
        sequence2 = srcs[1]
        sequence3 = srcs[2]
        assert len(sequence1) == len(sequence2)
        assert len(sequence1) == len(sequence3)

        sequence_length = len(sequence1)
        outname = os.path.join(options.output_dirname, 'original_sequence')

        if options.verbose:
            print "Writing file readable by ZipHMM to '%s'..." % outname,
            sys.stdout.flush()

        seen = set()
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        with open(outname, 'w', 64 * 1024) as f:
            for i in xrange(sequence_length):
                s1, s2, s3 = sequence1[i].upper(), sequence2[i].upper(), sequence3[i].upper()
                seen.add(s1)
                seen.add(s2)
                seen.add(s3)

                if s1 in clean and s2 in clean and s3 in clean:
                    i1, i2, i3 = nuc_map[s1], nuc_map[s2], nuc_map[s3]
                    print >> f, i1 + 4*i2 + 16*i3,
                else:
                    print >> f, 64,

        if options.verbose:
            print "done"
        if len(seen - set('ACGTN-')) > 1:
            print >> sys.stderr, "I didn't understand the following symbols form the input sequence: %s" % (
                ''.join(list(seen - set('ACGTN-'))))

        if options.verbose:
            print "ZipHMM is pre-processing...",
            sys.stdout.flush()
        f = Forwarder.fromSequence(seqFilename=outname, alphabetSize=65, minNoEvals=500)
        if options.verbose:
            print "done"

    elif len(names) == 4:
        # QUARTET ALIGNMENT ###########################################################################
        if options.verbose:
            print "Assuming quartet alignment between '%s', '%s', '%s', and '%s'" % (names[0], names[1], names[2], names[3])
        srcs = [alignments[name].seq for name in names]

        os.mkdir(options.output_dirname)
        sequence1 = srcs[0]
        sequence2 = srcs[1]
        sequence3 = srcs[2]
        sequence4 = srcs[3]
        assert len(sequence1) == len(sequence2)
        assert len(sequence1) == len(sequence3)
        assert len(sequence1) == len(sequence4)

        sequence_length = len(sequence1)
        outname = os.path.join(options.output_dirname, 'original_sequence')

        if options.verbose:
            print "Writing file readable by ZipHMM to '%s'..." % outname,
            sys.stdout.flush()

        seen = set()
        nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        with open(outname, 'w', 64 * 1024) as f:
            for i in xrange(sequence_length):
                s1, s2, s3, s4 = sequence1[i].upper(), sequence2[i].upper(), sequence3[i].upper(), sequence4[i].upper()
                seen.add(s1)
                seen.add(s2)
                seen.add(s3)
                seen.add(s4)

                if s1 in clean and s2 in clean and s3 in clean and s4 in clean:
                    i1, i2, i3, i4 = nuc_map[s1], nuc_map[s2], nuc_map[s3], nuc_map[s4]
                    print >> f, i1 + 4*i2 + 16*i3 + 32*i4,
                else:
                    print >> f, 128,

        if options.verbose:
            print "done"
        if len(seen - set('ACGTN-')) > 1:
            print >> sys.stderr, "I didn't understand the following symbols form the input sequence: %s" % (
                ''.join(list(seen - set('ACGTN-'))))

        if options.verbose:
            print "ZipHMM is pre-processing...",
            sys.stdout.flush()
        f = Forwarder.fromSequence(seqFilename=outname, alphabetSize=9, minNoEvals=500)
        if options.verbose:
            print "done"

    else:
        print 'There are', len(names), 'species identified. We do not know how to convert that into something'
        print 'that CoalHMM can handle, sorry.'
        sys.exit(1)

    if options.verbose:
        print "Writing ZipHMM data to '%s'..." % options.output_dirname,
        sys.stdout.flush()
        
    f.writeToDirectory('/'.join(options.output_dirname.split("/")[options.where_path_ends:]))
    if options.verbose:
        print "done"


if __name__ == "__main__":
    main()
