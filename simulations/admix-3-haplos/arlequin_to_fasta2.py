#!/usr/bin/env python

import sys

assert 4 == len(sys.argv), 'Expected one argument for path to file in Arlequin format.'
total_sites = int(sys.argv[2])

class Reader:
    def __init__(self, path):
        self.index = 0
        with open(path, 'r') as f:
            self.lines = f.read().splitlines()

    @property
    def end_of_file(self):
        return self.index >= len(self.lines)

    def read_line(self):
        line = self.lines[self.index]
        self.index += 1
        return line

    def skip_lines(self, n):
        self.index += n


def main():
    reader = Reader(sys.argv[1])
    prefix=sys.argv[3]
    print prefix
    cc = sys.argv[1].split('.')[-2].split('_')[-1]
    print "cc: "+str(cc)
    printcounter=1

    while not reader.end_of_file:
	#this takes the curser down to the line just before the number of the polymorphic sites.
        if '#Total number of polymorphic sites: ' not in reader.read_line():
            continue

        reader.skip_lines(1)
        sites = frozenset(map(int, reader.read_line()[1:].split(',')))
        sample_number = 0

        while not reader.end_of_file:
            line = reader.read_line() 
            if 'SampleName' not in line:
                continue

            sample_number += 1
	    number_in_sample_number=0
            sample_name = line.split('"')[1]
            sample_data = ''
	    found_sample_data=False

            while not reader.end_of_file:
                line = reader.read_line()
		print line

                if '}' in line:
		    print "found '}' in line"
                    break



                if 'SampleData' in line:
                    found_sample_data=True
		    line= reader.read_line()
		if not found_sample_data:
		    continue
		    
                fields = line.split('\t')
                if len(fields) != 3:
                    continue
		number_in_sample_number+=1
		if prefix:
			sample_path = prefix+'/sample_{0}_{1}.fasta'.format(sample_number, number_in_sample_number)
		else:
			sample_path = 'sample_{0}_{1}.fasta'.format(sample_number, number_in_sample_number)

                sample_data += fields[2].strip()

                

            	print 'Saving {0}...'.format(sample_path)

            	with open(sample_path, 'w') as f:
                	f.write('> {0}\n'.format(printcounter))
			printcounter+=1

                	j = 0
                	# for i in xrange(max(sites) + 1):
                	for i in xrange(total_sites):
                    		if i in sites:
                        		f.write(sample_data[j])
                        		j += 1
                    		else:
                        		f.write('a')

                    		if 0 == (i + 1) % 100:
                        		f.write('\n')

                	f.write('\n')

        

main()
