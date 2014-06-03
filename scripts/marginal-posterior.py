#!/usr/bin/env python

"""

Script for marginalising posterior decoding tables so we can get a summary for these.

"""

from argparse import ArgumentParser
from scipy import array

def main():
    """
    Run the main script.
    """
    usage = """%(prog)s [options] infile

This program reads a posterior decoding table or tables and outputs the marginal posteriors, i.e.
the distribution for each state averaged over the table(s).

"""

    parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/dev/stdout",
                        help="Output file for the estimate (/dev/stdout)")

    parser.add_argument('infile', nargs='?', help='Input file (/dev/stdin)', default='/dev/stdin')

    options = parser.parse_args()

    metadata = []
    posterior_counts = {}
    with open(options.infile) as infile:
        for line in infile:
            if line.startswith('##'):
                # Meta information we might want to keep. We keep all except Sequence which we
                # don't have any use for here.
                if not line.startswith('## Sequence:'):
                    metadata.append(line.strip())

            elif line.startswith('#'):
                # Leave this as a way of putting comments in the file
                pass

            else:
                seqname, pos, posteriors = line.split('\t', 2)
                posteriors = map(float, posteriors.split())
                if seqname in posterior_counts:
                    posterior_counts[seqname] += posteriors
                else:
                    posterior_counts[seqname] = array(posteriors)


    with open(options.outfile, 'w') as outfile:
        for line in metadata:
            print >> outfile, line

        for seqname, posterior in posterior_counts.iteritems():
            print >> outfile, seqname, ' '.join(map(str, posterior/posterior.sum()))

if __name__ == '__main__':
    main()
