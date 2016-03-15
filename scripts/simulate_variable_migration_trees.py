#!/usr/bin/env python

from IMCoalHMM.emissions import coalescence_points
from argparse import ArgumentParser
from numpy import array
from numpy.random import sample
from random import randrange,random

from math import log, exp
from mcmc2 import ExpLogNormPrior, LogNormPrior
from variable_migration_model2 import VariableCoalAndMigrationRateModel

from multiprocessing import Pool

def printPyZipHMM(Matrix):
    finalString=""
    for i in range(Matrix.getHeight()):
        for j in range(Matrix.getWidth()):
            finalString=finalString+" "+str(Matrix[i,j])
        finalString=finalString+"\n"
    return finalString

def simulateHMMemission(trans_prob, init_prob, emission_prob, length=1000000):

    differences=[]
    print array(init_prob).cumsum()
    simTime=array(init_prob).cumsum().searchsorted(sample(1))[0]
    differences.append(treeHeightToDiff(emission_prob,simTime))
    for _ in xrange(length-1):
        simTime=array(trans_prob[simTime,:]).cumsum().searchsorted(sample(1))[0]
        differences.append(treeHeightToDiff(emission_prob,simTime))
    return differences

def simulateHMMemissionWrap(inp):
    if len(inp)==4:
        return simulateHMMemission(inp[0],inp[1],inp[2],length=inp[3])
    return simulateHMMemission(inp[0],inp[1],inp[2])
        
    
def treeHeightToDiff(emission_prob, index):
    if random()<emission_prob[index,1]: #emission_prob[index,1] is the probability in time period index leads to alignment
        return 1 
    return 0

def makeHMMalignment(difference11, difference12, difference22,filename='/home/svendvn'):
    str1="1         "
    str2="2         "
    str3="3         "
    str4="4         "
    seq_length=len(difference11)
    for i in xrange(seq_length):
        letters=adds(difference11[i],difference12[i],difference22[i])
        str1+=letters[0]
        str2+=letters[1]
        str3+=letters[2]
        str4+=letters[3]
    with open(filename+"/alignment.phylip",'w') as fil:
        fil.write(" 4 "+str(seq_length)+"\n")
        fil.write(str1+"\n")
        fil.write(str2+"\n")
        fil.write(str3+"\n")
        fil.write(str4+"\n")


def simulate(trans_probs, init_probs, break_points, coal_rates=1000.0, length=1000000, filename='/home/svendvn/', simAlign=False, subsRate=4*25*20000*1e-9):
    ''' 
    trans_probs and init_probs contain 3 matrices each. break_points should include 0 but not infinity
    '''
    simTimes=[0]*3
    
    
    #simulating the initial times
    for i in range(3):
        simTimes[i]=array(init_probs[i]).cumsum().searchsorted(sample(1))[0]
        
    #makes coalescence points from breakpoints
    coalPoints1=coalescence_points(break_points,coal_rates)
    coalPoints2=[i/subsRate for i in coalPoints1]
    
    #getting the newickformat of the first tree. 
    stringToWrite=simTreeFromPoints(simTimes,coalPoints2)
    
    #simulates the next many trees
    reps=0
    
    #print printPyZipHMM(trans_probs[0])
    trans_probs=translateToArray(trans_probs)
    #print trans_probs
    
    if simAlign:
        str1=""
        str2=""
        str3=""
        str4=""
        with open(filename+"trees.nwk", 'w') as fil:
            for _ in xrange(length):
                c12=coalPoints1[simTimes[0]]
                c13=coalPoints1[simTimes[1]]
                c34=coalPoints1[simTimes[2]]
                t12=random()<jk(c12)
                t13=random()<jk(c13)
                t34=random()<jk(c34)
                addOns=adds(t12,t13,t34)
                str1+=addOns[0]
                str2+=addOns[1]
                str3+=addOns[2]
                str4+=addOns[3]
                reps+=1
                change=False
                for i in range(3):
                    new=array(trans_probs[i][simTimes[i],:]).cumsum().searchsorted(sample(1))[0]
                    if new != simTimes[i]:
                        simTimes[i]=new
                        change=True
                if change:
                    stringToWrite="["+str(reps)+"]"+stringToWrite
                    fil.write(stringToWrite+"\n")
                    stringToWrite=simTreeFromPoints(simTimes,coalPoints2)
                    reps=0
            #inserting the last line
            stringToWrite="["+str(reps)+"]"+stringToWrite
            fil.write(stringToWrite+"\n")
        print str1.count("C")
        print str2.count("C")
        print str3.count("C")
        print str4.count("C")
        with open(filename+'alignment.phylip',"w") as fil:
            fil.write(" 4 "+str(length)+"\n")
            fil.write("1         "+str1+"\n")
            fil.write("2         "+str2+"\n")
            fil.write("3         "+str3+"\n")
            fil.write("4         "+str4+"\n")
    else:
        with open(filename+"trees.nwk", 'w') as fil:
            for _ in xrange(length):
                #this variable is True if we make a change, that is jumps to another tree
                reps+=1
                change=False
                for i in range(3):
                    new=array(trans_probs[i][simTimes[i],:]).cumsum().searchsorted(sample(1))[0]
                    #print new
                    if new != simTimes[i]:
                        simTimes[i]=new
                        change=True
                if change:
                    stringToWrite="["+str(reps)+"]"+stringToWrite
                    print stringToWrite
                    fil.write(stringToWrite+"\n")
                    stringToWrite=simTreeFromPoints(simTimes,coalPoints2)
                    reps=0
            #inserting the last line
            stringToWrite="["+str(reps)+"]"+stringToWrite
            print stringToWrite
            fil.write(stringToWrite+"\n")
            
def jk(t):
    #returns probability of unequal:
    return 0.75-0.75*exp(-4.0/3.0*2*t)

def adds(n1,n2,n3):
    if not n1 and not n2 and not n3:
        return ("A","A","A","A")
    else:
        if not n1 and not n2 and n3:
            return ("A","A","A","C")
        elif n1 and not n2 and not n3:
            return ("A","C","A","A")
        elif not n1 and n2 and not n3:
            return ("A","A","C","C")
        elif n1 and n2 and not n3:
            return ("C","A","A","A")
        elif n1 and not n2 and n3:
            return ("A","C","A","C")
        elif not n1 and n2 and n3:
            return ("A","A","C","A")
        elif n1 and n2 and n3:
            return ("A","C","C","A")
        else:
            assert False, "Woops"
           
def translateToArray(trans_probs):
    res=[]
    for element in trans_probs:
        shell=[]
        for i in range(element.getHeight()):
            shell.append([0]*element.getWidth())
        for i in range(element.getHeight()):
            for j in range(element.getWidth()):
                shell[i][j]=float(element[i,j])
        res.append(array(shell))
    return res
    

    

#((3:0.261,4:0.261):1.682,(1:0.568,2:0.568):1.376)
#(2:2.715,(1:0.956,(3:0.511,4:0.511):0.445):1.759);
def simTreeFromPoints(times, coalPoints, names=["1","2","3","4"]):
    c12=coalPoints[times[0]]
    c13=coalPoints[times[1]]
    c34=coalPoints[times[2]]
    if times[1]>times[0] and times[1]>times[2]:   #
        diff1=c13-c12
        diff2=c13-c34
        res=("(("+names[0]+":"+str(c12)+","+names[1]+":"+str(c12)+"):" +str(diff1) + ",("+names[2]+":"
             +str(c34)+","+names[3]+":"+str(c34)+ "):" +str(diff2)+")")
    elif times[0]>times[1] and times[0]>times[2]:
        if times[1]==times[2]:
            substring="("+names[2]+":"+str(c13)+",("+names[0]+":"+str(c13/2)+","+names[3]+":"+str(c13/2)+"):"+str(c13/2)+"):"
        elif times[2]>times[1]:
            substring="("+names[3]+":"+str(c34)+",("+names[0]+":"+str(c13)+","+names[2]+":"+str(c13)+"):"+str(c34-c13)+"):"
        else:
            substring="("+names[0]+":"+str(c13)+",("+names[2]+":"+str(c34)+","+names[3]+":"+str(c34)+"):"+str(c13-c34)+"):"
        res=("("+names[1]+":"+str(c12)+","+substring+str(c12-max(c13,c34))+")")
    elif times[2]>times[1] and times[2]>times[0]:
        times.reverse()
        namesTwo=[names[2],names[3],names[0],names[1]]
        return simTreeFromPoints(times,coalPoints, namesTwo)
    elif times[0]==times[1] and times[0]==times[2]:
        res=("(("+names[0]+":"+str(c12/2)+","+names[3]+":"+str(c12/2)+"):" +str(c12/2) + ",("+names[1]+":"
             +str(c34/2)+","+names[2]+":"+str(c34/2)+ "):" +str(c34/2)+")")
    elif times[0]==times[1]:
        de=randrange(3)
        if de==0:
            c24=(c13-c34)/2+c34
            maxc=c24
            substring="("+names[1]+":"+str(c24)+",("+names[2]+":"+str(c34)+","+names[3]+":"+str(c34)+"):"+str(c24-c34)+"):"
        elif de==1:
            c24=c34/2
            maxc=c34
            substring="("+names[3]+":"+str(c34)+",("+names[1]+":"+str(c24)+","+names[2]+":"+str(c24)+"):"+str(c34-c24)+"):"
        else:
            c24=c34/2
            maxc=c34
            substring="("+names[2]+":"+str(c34)+",("+names[1]+":"+str(c24)+","+names[3]+":"+str(c24)+"):"+str(c34-c24)+"):"
        res=("("+names[0]+":"+str(c12)+","+substring+str(c12-maxc)+")")
    elif times[0]==times[2]:
        c24=c12/2
        res=("(("+names[0]+":"+str(c13)+","+names[2]+":"+str(c13)+"):" +str(c12-c13) + ",("+names[1]+":"
             +str(c24)+","+names[3]+":"+str(c24)+ "):" +str(c12-c24)+")")
    elif times[2]==times[1]:
        times.reverse()
        namesTwo=[names[2],names[3],names[0],names[1]]
        return simTreeFromPoints(times,coalPoints, namesTwo)
    else:
        assert False, "We should never reach this point!"
        
    return res

def main():
    
    usage = """%(prog)s [options] <forwarder dirs>
    """
    
    parser = ArgumentParser(usage=usage, version="%(prog)s 1.0")

    parser.add_argument("-o", "--outfile",
                        type=str,
                        default="/home/svendvn",
                        help="Output file for the estimate (/dev/stdout)")
    
    parser.add_argument("-t", '--type', type=int, default=0, help='should we simulate trees using (0) or alignments(1)')
    parser.add_argument('--breakpoints_time', default=1.0, type=float, help='this number moves the breakpoints up and down. Smaller values will give sooner timeperiods.')
    parser.add_argument('--intervals', nargs='+', default=[5,5,5,5], type=int, help='This is the setup of the intervals. They will be scattered equally around the breakpoints')
    parser.add_argument('--seq_length', default=1000000, type=int, help='This is the setup of the intervals. They will be scattered equally around the breakpoints')
    parser.add_argument('--breakpoints_tail_pieces', default=0, type=int, help='this produce a tail of last a number of pieces on the breakpoints')
    parser.add_argument('--paramsElaborate', nargs='+', default=[], type=float, help='should the initial step be the initial parameters(otherwise simulated from prior).')
    parser.add_argument('--fix_time_points', nargs='+',default=[], help='this will fix the specified time points. Read source code for further explanation')



    optimized_params = [
        ('theta', 'effective population size in 4Ne substitutions', 2e6 / 1e9),
        ('rho', 'recombination rate in substitutions', 0.4),
        ('migration-rate', 'migration rate in number of migrations per substitution', 250.0),
        ('Ngmu4', 'substitutions per 4Ng years', 4*20000*25*1e-9) #it is only used when we use trees as input data. 
    ]

    for parameter_name, description, default in optimized_params:
        parser.add_argument("--%s" % parameter_name,
                            type=float,
                            default=default,
                            help="Initial guess at the %s (%g)" % (description, default))
        
        
    options = parser.parse_args()
    
    
    theta=options.theta
    rho=options.rho
    
    init_coal = 1 / (theta / 2)
    init_mig = options.migration_rate
    init_recomb = rho
    intervals=options.intervals
    no_epochs = len(intervals)
    
    trueParams=[init_coal]*(2*no_epochs)+[init_mig]*(2*no_epochs)+[init_recomb]
    if options.paramsElaborate:
        assert len(options.paramsElaborate)==len(trueParams), "elaborate parameters not of correct length. It should be len(elaborateParameters)=4*len(intervals)+1."
        trueParams=options.paramsElaborate

    # FIXME: I don't know what would be a good choice here...
    # intervals = [4] + [2] * 25 + [4, 6]

    def fix_scaler():
        return fixed_time_points
    if options.fix_time_points:
        fixed_time_points=[(int(f),float(t)) for f,t in zip(options.fix_time_points[::2],options.fix_time_points[1::2])]
        fixed_time_pointer=fix_scaler
    else:
        fixed_time_pointer=None
    

    def transform(parameters):
        coal_rates_1 = tuple(parameters[0:no_epochs])
        coal_rates_2 = tuple(parameters[no_epochs:(2 * no_epochs)])
        mig_rates_12 = tuple(parameters[(2 * no_epochs):(3 * no_epochs)])
        mig_rates_21 = tuple(parameters[(3 * no_epochs):(4 * no_epochs)])
        recomb_rate = parameters[-1]
        theta_1 = tuple([2 / coal_rate for coal_rate in coal_rates_1])
        theta_2 = tuple([2 / coal_rate for coal_rate in coal_rates_2])
        return theta_1 + theta_2 + mig_rates_12 + mig_rates_21 + (recomb_rate,)
    
    
    # load alignments
    models=[]
    models.append(VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_11, intervals, breaktimes=options.breakpoints_time, breaktail=options.breakpoints_tail_pieces,time_modifier=fixed_time_pointer))
    models.append(VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_12, intervals, breaktimes=options.breakpoints_time, breaktail=options.breakpoints_tail_pieces,time_modifier=fixed_time_pointer))
    models.append(VariableCoalAndMigrationRateModel(VariableCoalAndMigrationRateModel.INITIAL_22, intervals, breaktimes=options.breakpoints_time, breaktail=options.breakpoints_tail_pieces,time_modifier=fixed_time_pointer))
    
    trans_probs=[]
    init_probs=[]
    emiss_probs=[]
    for i in range(3):
        inp,trp,emp,bre=models[i].build_hidden_markov_model(trueParams)
        trans_probs.append(trp)
        init_probs.append(inp)
        emiss_probs.append(emp)
    print "break points:"
    print bre
    
    trans_probs=translateToArray(trans_probs)
    emiss_probs=translateToArray(emiss_probs)
        
    #print bre
    
    if options.type==0:
        simulate(filename=options.outfile, break_points=bre, trans_probs=trans_probs, length=options.seq_length, init_probs=init_probs, simAlign=False, subsRate=options.Ngmu4)
    elif options.type==1:
        diffs=[]
        for n in range(3):
            diffs.append(simulateHMMemission(trans_probs[n], init_probs[n], emiss_probs[n],length=options.seq_length))
        makeHMMalignment(diffs[0],diffs[1],diffs[2],filename=options.outfile)
    #else:
    #    simulate(filename=options.outfile, break_points=bre, trans_probs=trans_probs, length=options.seq_length, init_probs=init_probs, simAlign=False, subsRate=options.Ngmu4)
    
if __name__ == '__main__':
    main()
    print simTreeFromPoints([1,1,1],[1.0,2.0,3.0], names=["A","B","C","D"])
    print ""
    
    print simTreeFromPoints([1,1,0],[1.0,2.0,3.0], names=["A","B","C","D"])
    print simTreeFromPoints([0,1,1],[1.0,2.0,3.0], names=["A","B","C","D"])
    print simTreeFromPoints([1,0,1],[1.0,2.0,3.0], names=["A","B","C","D"])
    print ""
    
    print simTreeFromPoints([1,0,0],[1.0,2.0,3.0], names=["A","B","C","D"])
    print simTreeFromPoints([0,1,0],[1.0,2.0,3.0], names=["A","B","C","D"])
    print simTreeFromPoints([0,0,1],[1.0,2.0,3.0], names=["A","B","C","D"])
    print ""
    
    print simTreeFromPoints([0,2,1],[1.0,2.0,3.0], names=["A","B","C","D"])
    print simTreeFromPoints([0,1,2],[1.0,2.0,3.0], names=["A","B","C","D"])
    print simTreeFromPoints([2,0,1],[1.0,2.0,3.0], names=["A","B","C","D"])
    print simTreeFromPoints([1,0,2],[1.0,2.0,3.0], names=["A","B","C","D"])
    print simTreeFromPoints([2,1,0],[1.0,2.0,3.0], names=["A","B","C","D"])
    print simTreeFromPoints([1,2,0],[1.0,2.0,3.0], names=["A","B","C","D"])
    
    
    
    
    

    