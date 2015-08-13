import subprocess
from argparse import ArgumentParser

parser = ArgumentParser(usage="takes a file with parameters either isolation og isolation-migration and simulates from ms", version="%(prog)s 1.0")


parser.add_argument("--filename",
                        type=str,
                        default=None,
                        help="A file that contains the line of headers and a line of parameters")

parser.add_argument("--treefile",
                        type=str,
                        default=None,
                        help="A file that is going to contain the trees in nwk format")
parser.add_argument("--seqfile",
                        type=str,
                        default=None,
                        help="A file that is going to contain the sequences in phylip format")

parser.add_argument("--seg_length",
                        type=int,
                        default=100000,
                        help="The length of sequence(for ms)")
#parser.add_argument("--Ne",
#			type=float,
#			default=20000,
#			help="The effective population size")
#parser.add_argument("--gen_years",
#			type=float,
#			default=25,
#			help="the generation size")
#parser.add_argument("--mu",#
#			type=float,
#			default=1e-9,
#			help="the generation size")
			

options=parser.parse_args()

lastAfterHyphen=options.filename.split("-")[-1]
typeOf=lastAfterHyphen.split(".")[0]


with open(options.filename, 'r') as f:
	f.readline()
	line=f.readline()
	parameters=map(float,line.split("\t"))

if typeOf=="isolation":
	thetaYearsMu=parameters[1]
	rhoPerSubsPerSite=parameters[2]
	rhoPerGenerationPerGenome=rhoPerSubsPerSite*options.seg_length*thetaYearsMu
	splitTimeSubs=parameters[0]
	splitTimeMS=splitTimeSubs/thetaYearsMu
	subprocess.call("ms 4 1 -T -r "+str(rhoPerGenerationPerGenome)+" "+str(options.seg_length)+" "+
			"-I 2 2 2 0.0 -ej "+ str(splitTimeMS)+" 2 1  | tail -n +4 | grep -v // > "+options.treefile,shell=True)
if typeOf=="migration":
	thetaYearsMu=parameters[2]
	rhoPerSubsPerSite=parameters[3]
	rhoPerGenerationPerGenome=rhoPerSubsPerSite*options.seg_length*thetaYearsMu
	migrationsPerSubsPerSite=parameters[4]
	migrationsParameter=migrationsPerSubsPerSite*thetaYearsMu
	splitTimeSubs=parameters[0]
	splitTimeMS=spitTimeSubs/thetaYearsMu
	migrationTimeStartSubs=parameters[1]
	migtationTimeStartMS=migrationTimeStartSubs/thetaYearsMu
	subprocess.call("ms 4 1 -T -r "+str(rhoPerGenerationPerGenome)+" "+str(options.seg_length)+" "+
			"-I 2 2 2 0.0 -em "+ str(splitTimeMS)+" 1 2 "+ str(migrationsParameter)+" -em "+str(splitTimeMS)+" 2 1 "+ str(migrationsParameter)+
		   	" -ej " + str(migrationTimeStartMS)+" 2 1 | tail -n +4 | grep -v // > "+options.treefile,shell=True)

subprocess.call("seq-gen -q -mHKY -l "+str(options.seg_length)+" -s "+str(thetaYearsMu)+" -p "+
	str(options.seg_length/10)+" < "+options.treefile+" > "+options.seqfile, shell=True)


	
	
