#!bin/bash

source activate my_root

echo "fil startet" 

echo "========= Job started  at `date` =========="

echo $SLURM_JOBID

simulateWith=$1
analyseWith=$2

cd /home/svendvn/IMCoalHMM/scripts
cp *py /scratch/$SLURM_JOBID
cd /scratch/$SLURM_JOBID

echo "copied files"

no_sims=2
no_chunks=100 # FIXME

Ne=20000
gen=25
mu=10^-9
rho_per_gen=$(bc -l <<< 0.01/10^6)
seg_length=1000000
coal_mig_rate=0.5

sym_coal_mig_rate=$(bc -l <<< 2*${coal_mig_rate}) # times two because ms divides by number of populations

theta_years=$(( 4*$Ne*$gen ))
coal_rho=$(bc -l <<< 4*$Ne*$rho_per_gen*$seg_length)

# units in substitutions
theta_subs=$(bc -l <<< $theta_years*$mu )
rho_subs=$(bc -l <<< $rho_per_gen/$gen/$mu )

substime_for_change=0.0005 #the unit is in substitutions
substime_for_change2=0.0010
substime_for_change_back2=$(bc -l <<< ${substime_for_change2}*3)

mstime_for_change=$(bc -l <<< $substime_for_change/$theta_subs) #this unit is in 4Ng, that is the time of one generation times 4N
mstime_for_change2=$(bc -l <<< $substime_for_change/$theta_subs)
changed_migration=$(bc -l <<< ${sym_coal_mig_rate}*0.5)
changed_migration2=$(bc -l <<< ${sym_coal_mig_rate}*0.2)
mstime_for_change_back=$(bc -l <<< ${mstime_for_change}*2)
mstime_for_change_back2=$(bc -l <<< ${mstime_for_change2}*3)
initial_substime_for_change=$(bc -l <<< ${substime_for_change}*0.21)
initial_substime_for_change_back=$(bc -l <<< ${substime_for_change2}*0.21)
initial_substime_for_change_back2=$(bc -l <<< ${substime_for_change_back2}*0.21)
initial_substime_for_change2=$(bc -l <<< ${substime_for_change2}*0.21)

for sim in `eval echo {1..${no_sims}}`; do
    
	num=$RANDOM
		mkdir /scratch/$SLURM_JOBID/IMCoalHMM-simulations.$num -p
	simdir=/scratch/$SLURM_JOBID/IMCoalHMM-simulations.$num
    treefile=${simdir}/trees.nwk
    seqfile=${simdir}/alignment.phylip

	echo "directories made"
    
    for chunk in `eval echo {1..${no_chunks}}`; do
        ziphmmfile12=${simdir}/alignment.$chunk.12.ziphmm
        ziphmmfile13=${simdir}/alignment.$chunk.13.ziphmm
        ziphmmfile14=${simdir}/alignment.$chunk.14.ziphmm
        ziphmmfile23=${simdir}/alignment.$chunk.23.ziphmm
        ziphmmfile24=${simdir}/alignment.$chunk.24.ziphmm
        ziphmmfile34=${simdir}/alignment.$chunk.34.ziphmm
	
	if [ "$simulateWith" = "A" ]
	then
		echo "simulate from A"
		ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 -em ${mstime_for_change} 1 2 $changed_migration -ej $mstime_for_change_back 1 2 | tail -n +4 | grep -v // > ${treefile}
		out1="ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 -em ${mstime_for_change} 1 2 $changed_migration -ej $mstime_for_change_back 1 2 | tail -n +4 | grep -v // > ${treefile}"
        	index=Asided
	elif [ "$simulateWith" = "B" ]
	then
		echo "simulate from B"
		out1="		ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 -em ${mstime_for_change} 2 1 $changed_migration -ej $mstime_for_change_back 1 2 | tail -n +4 | grep -v // > ${treefile}"
		ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 -em ${mstime_for_change} 2 1 $changed_migration -ej $mstime_for_change_back 1 2 | tail -n +4 | grep -v // > ${treefile}
		index=Bsided
	elif [ "$simulateWith" = "E" ]
	then
		echo "simulate from E"
		ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 -em ${mstime_for_change} 1 2 $changed_migration -em ${mstime_for_change} 2 1 $changed_migration -ej $mstime_for_change_back 1 2 | tail -n +4 | grep -v // > ${treefile}
		out1="		ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 -em ${mstime_for_change} 1 2 $changed_migration -em ${mstime_for_change} 2 1 $changed_migration -ej $mstime_for_change_back 1 2 | tail -n +4 | grep -v // > ${treefile}"
		index=Esided
	else
		echo "simulate from T"
		ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 -em ${mstime_for_change} 1 2 $changed_migration -em ${mstime_for_change} 2 1 $changed_migration2 -ej $mstime_for_change_back 1 2 | tail -n +4 | grep -v // > ${treefile}
		out1="		ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 -em ${mstime_for_change} 1 2 $changed_migration -em ${mstime_for_change} 2 1 $changed_migration2 -ej $mstime_for_change_back 1 2 | tail -n +4 | grep -v // > ${treefile}"
		index=Tsided
	fi		
	seq-gen -q -mHKY -l ${seg_length} -s ${theta_subs} -p $(( $seg_length / 10 )) < ${treefile} > ${seqfile}
	echo "data made"
        python /scratch/$SLURM_JOBID/prepare-alignments.py --names=1,2 ${seqfile} phylip ${ziphmmfile12} --where_path_ends 3
        python /scratch/$SLURM_JOBID/prepare-alignments.py --names=1,3 ${seqfile} phylip ${ziphmmfile13} --where_path_ends 3
        python /scratch/$SLURM_JOBID/prepare-alignments.py --names=1,4 ${seqfile} phylip ${ziphmmfile14} --where_path_ends 3
        python /scratch/$SLURM_JOBID/prepare-alignments.py --names=2,3 ${seqfile} phylip ${ziphmmfile23} --where_path_ends 3
        python /scratch/$SLURM_JOBID/prepare-alignments.py --names=2,4 ${seqfile} phylip ${ziphmmfile24} --where_path_ends 3
        python /scratch/$SLURM_JOBID/prepare-alignments.py --names=3,4 ${seqfile} phylip ${ziphmmfile34} --where_path_ends 3
	echo "alignments made"
        
    done
    	
	
	out2= "ms 4 1 -T -r ${coal_rho} ${seg_length} -I 2 2 2 -em ${mstime_for_change} 1 2 $changed_migration -ej $mstime_for_change_back 1 2 | tail -n +4 | grep -v // > ${treefile}"
	out1="	out1="	python /scratch/$SLURM_JOBID/variable-migration-model-mcmc.py -o INMmcmc-100d-mc3_15adap1-40states_single_scaling-${sim}-chain.txt -a11 ${simdir}/*.11.ziphmm -a12 ${simdir}/*.12.ziphmm -a22 ${simdir}/*.22.ziphmm --samples 1000 --thinning 20  --sd_multiplyer 0.1  --adap 1 --mc3 --parallels 15 --mc3_fixed_temp_max 800 --mc3_flip_suggestions 40  --mc3_sort_chains --migration_uniform_prior 3000 --breakpoints_tail_pieces 5 --intervals 10 10 10 10 --fix_time_points 9 ${initial_substime_for_change} 19 ${initial_substime_for_change_back} "


        if [ "$analyseWith" = "A" ]
        then
		nameOfFile=INM-100d-6align-${index}_Asided-neldermead-${sim}.txt
               	python /scratch/$SLURM_JOBID/variable-migration-model-6-align.py -o $nameOfFile -a12 ${simdir}/*.12.ziphmm -a13 ${simdir}/*.13.ziphmm -a23 ${simdir}/*.23.ziphmm -a14 ${simdir}/*.14.ziphmm -a24 ${simdir}/*.24.ziphmm -a34 ${simdir}/*.34.ziphmm --no_mcmc \
			--startWithGuessElaborate 2000 2000 2000 2000 2000 2000 0.0 500.0 0.0 0.0 0.0 0.0 0.4 --fix_params 6 8 9 10 11 --fix_params_to_be_equal=0,1,2,3,4,5 \
			--startWithGuess --breakpoints_tail_pieces 5 --intervals 10 10 10 --fix_time_points 10 ${substime_for_change} 20 ${substime_for_change2} 
		out2="python /scratch/$SLURM_JOBID/variable-migration-model-6-align.py -o $nameOfFile -a12 ${simdir}/*.12.ziphmm -a13 ${simdir}/*.13.ziphmm -a23 ${simdir}/*.23.ziphmm -a14 ${simdir}/*.14.ziphmm -a24 ${simdir}/*.24.ziphmm -a34 ${simdir}/*.34.ziphmm --no_mcmc --startWithGuessElaborate 2000 2000 2000 2000 2000 2000 0.0 500.0 0.0 0.0 0.0 0.0 0.4 --fix_params 6 8 9 10 11 --fix_params_to_be_equal=0,1,2,3,4,5 --startWithGuess --breakpoints_tail_pieces 5 --intervals 10 10 10 --fix_time_points 10 ${substime_for_change} 20 ${substime_for_change2} "
        elif [ "$analyseWith" = "B" ]
        then
                nameOfFile=INM-100d-6align-${index}_Bsided-neldermead-${sim}.txt
               	python /scratch/$SLURM_JOBID/variable-migration-model-6-align.py -o $nameOfFile -a12 ${simdir}/*.12.ziphmm -a13 ${simdir}/*.13.ziphmm -a23 ${simdir}/*.23.ziphmm \ 
			-a14 ${simdir}/*.14.ziphmm -a24 ${simdir}/*.24.ziphmm -a34 ${simdir}/*.34.ziphmm --no_mcmc \
                        --startWithGuessElaborate 2000 2000 2000 2000 2000 2000 0.0 0.0 0.0 0.0 500.0 0.0 0.4 --fix_params 6 7 8 9 11 --fix_params_to_be_equal=0,1,2,3,4,5 \
                        --startWithGuess --breakpoints_tail_pieces 5 --intervals 10 10 10 --fix_time_points 10 ${substime_for_change} 20 ${substime_for_change2}
		out2="python /scratch/$SLURM_JOBID/variable-migration-model-6-align.py -o $nameOfFile -a12 ${simdir}/*.12.ziphmm -a13 ${simdir}/*.13.ziphmm -a23 ${simdir}/*.23.ziphmm \ 
                        -a14 ${simdir}/*.14.ziphmm -a24 ${simdir}/*.24.ziphmm -a34 ${simdir}/*.34.ziphmm --no_mcmc \
                        --startWithGuessElaborate 2000 2000 2000 2000 2000 2000 0.0 0.0 0.0 0.0 500.0 0.0 0.4 --fix_params 6 7 8 9  11 --fix_params_to_be_equal=0,1,2,3,4,5 \
                        --startWithGuess --breakpoints_tail_pieces 5 --intervals 10 10 10 --fix_time_points 10 ${substime_for_change} 20 ${substime_for_change2}"
        elif [ "$analyseWith" = "E" ]
	then
                nameOfFile=INM-100d-6align-${index}_Esided-neldermead-${sim}.txt
                python /scratch/$SLURM_JOBID/variable-migration-model-6-align.py -o $nameOfFile -a12 ${simdir}/*.12.ziphmm -a13 ${simdir}/*.13.ziphmm -a23 ${simdir}/*.23.ziphmm \ 
                        -a14 ${simdir}/*.14.ziphmm -a24 ${simdir}/*.24.ziphmm -a34 ${simdir}/*.34.ziphmm --no_mcmc \
                        --startWithGuessElaborate 2000 2000 2000 2000 2000 2000 0.0 500.0 0.0 0.0 500.0 0.0 0.4 --fix_params 6 8 9 11 --fix_params_to_be_equal=0,1,2,3,4,5:7,10 \
                        --startWithGuess --breakpoints_tail_pieces 5 --intervals 10 10 10 --fix_time_points 10 ${substime_for_change} 20 ${substime_for_change2}
                out2="python /scratch/$SLURM_JOBID/variable-migration-model-6-align.py -o $nameOfFile -a12 ${simdir}/*.12.ziphmm -a13 ${simdir}/*.13.ziphmm -a23 ${simdir}/*.23.ziphmm \
                        -a14 ${simdir}/*.14.ziphmm -a24 ${simdir}/*.24.ziphmm -a34 ${simdir}/*.34.ziphmm --no_mcmc \
                        --startWithGuessElaborate 2000 2000 2000 2000 2000 2000 0.0 500.0 0.0 0.0 500.0 0.0 0.4 --fix_params 6 8 9 11 --fix_params_to_be_equal=0,1,2,3,4,5:7,10 \
                        --startWithGuess --breakpoints_tail_pieces 5 --intervals 10 10 10 --fix_time_points 10 ${substime_for_change} 20 ${substime_for_change2}"
	else
                nameOfFile=INM-100d-6align-${index}_Tsided-neldermead-${sim}.txt
                python /scratch/$SLURM_JOBID/variable-migration-model-6-align.py -o $nameOfFile -a12 ${simdir}/*.12.ziphmm -a13 ${simdir}/*.13.ziphmm -a23 ${simdir}/*.23.ziphmm \
                        -a14 ${simdir}/*.14.ziphmm -a24 ${simdir}/*.24.ziphmm -a34 ${simdir}/*.34.ziphmm --no_mcmc \
                        --startWithGuessElaborate 2000 2000 2000 2000 2000 2000 0.0 500.0 0.0 0.0 500.0 0.0 0.4 --fix_params 6 8 9 11 --fix_params_to_be_equal=0,1,2,3,4,5 \
                        --startWithGuess --breakpoints_tail_pieces 5 --intervals 10 10 10 --fix_time_points 10 ${substime_for_change} 20 ${substime_for_change2}
                out2="python /scratch/$SLURM_JOBID/variable-migration-model-6-align.py -o $nameOfFile -a12 ${simdir}/*.12.ziphmm -a13 ${simdir}/*.13.ziphmm -a23 ${simdir}/*.23.ziphmm \
                        -a14 ${simdir}/*.14.ziphmm -a24 ${simdir}/*.24.ziphmm -a34 ${simdir}/*.34.ziphmm --no_mcmc \
                        --startWithGuessElaborate 2000 2000 2000 2000 2000 2000 0.0 500.0 0.0 0.0 500.0 0.0 0.4 --fix_params 6 8 9 11 --fix_params_to_be_equal=0,1,2,3,4,5 \
                        --startWithGuess --breakpoints_tail_pieces 5 --intervals 10 10 10 --fix_time_points 10 ${substime_for_change} 20 ${substime_for_change2}"
        fi



	echo $out1 >> $nameOfFile
	echo $out2 >> $nameOfFile
	echo "vi har ${no_chunks}*${seg_length} basepar" >>  $nameOfFile

	cp /scratch/$SLURM_JOBID/$nameOfFile /home/svendvn/

done

echo "========= Job finished at `date` =========="
