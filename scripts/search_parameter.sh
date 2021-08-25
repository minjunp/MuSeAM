#!/bin/bash

#work_dir="/project/samee/minjun/museam_runs"
output_base_dir="outputs"
par_output_dir="pars"
fasta_file="./data/all_sequences.fa"
readout_file="./data/regression_readout.dat"
train_file="train.py"

if [ -d ${output_base_dir} ]
then
	rm -rf ${output_base_dir}
fi
#mkdir -p ${output_base_dir}
mkdir -p ${par_output_dir}

#exec_file="${work_dir}/parameter_search.sh"
#echo "#!/bin/bash" > ${exec_file}

#code="/project/samee/minjun/mpra/code/dummy.py"

index=0
for filters in 256 512
do
  for kernel_size in 12 16 20 24
  do
    for pool_type in Max
    do
      for regularizer in L_1
      do
		    for epochs in 30 35 40 45 50
		    do
		      for batch_size in 256 512
		      do
						for alpha in 0
						do
						  for beta in 0
						  do
					        let index=index+1
					        parameter_file_name="${par_output_dir}/parameters_"${index}".txt"
					        echo "filters "${filters} > ${parameter_file_name}
					        echo "kernel_size "${kernel_size} >> ${parameter_file_name}
					        echo "pool_type "${pool_type} >> ${parameter_file_name}
					        echo "regularizer "${regularizer} >> ${parameter_file_name}
					        echo "epochs "${epochs} >> ${parameter_file_name}
					        echo "batch_size "${batch_size} >> ${parameter_file_name}
									echo "alpha "${alpha} >> ${parameter_file_name}
									echo "beta "${beta} >> ${parameter_file_name}
					        #output_dir="${output_base_dir}/out_${index}"
					  	    #python ${train_file} ${fasta_file} ${readout_file} ${parameter_file_name} > ${output_base_dir}/${index}.txt
						  done
						done
		      done
		    done
      done
    done
  done
done
#chmod +x ${exec_file}
