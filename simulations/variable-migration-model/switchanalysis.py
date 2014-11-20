from argparse import ArgumentParser

parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")
parser.add_argument("-f", "--filename",type=str)

options=parser.parse_args()

filename=options.filename

fil=open(filename, 'r')

startCountingAccept=False
startCountingReject=True

for line in fil:
	if startCountingAccept:
		listi=line.split("\t")
		
