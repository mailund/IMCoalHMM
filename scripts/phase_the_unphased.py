from argparse import ArgumentParser
from Bio import SeqIO
from random import random
import os

parser = ArgumentParser(usage="Takes two phased genomes in fasta format and converts it into one sequence, iid-like", version="%(prog)s 1.0")


parser.add_argument("--names",
                        type=str,
                        default=None,
                        help="A comma-separated list of names to use from the source file")
parser.add_argument("--in_filename", type=str, help="Input file")
parser.add_argument("--output_name", type=str, help="Where to write the new file")

options = parser.parse_args()
inf = open(options.in_filename)
alignments = SeqIO.to_dict(SeqIO.parse(inf, "phylip"))
if options.names:
	names = options.names.split(',')
else:
	names = list(alignments.keys())


srcs = [alignments[name].seq for name in names]
outname=options.output_name
count=0
with open(outname, 'w') as f:
	print >> f, " "+str(len(names)/2)+" "+str(len(srcs[0]))

	for i in range(len(names)/2):
		count+=1
		sstr=str(count)+"         "
        	sequence1 = srcs[i*2+0]
        	sequence2 = srcs[i*2+1]
        	assert len(sequence1) == len(sequence2)
        	sequence_length = len(sequence1)
            	for i in xrange(sequence_length):
                	s1, s2 = sequence1[i].upper(), sequence2[i].upper()
                    
                	if random()<0.5:
                    		sstr+=s1
                	else:
                    		sstr+=s2
		print >> f, sstr


