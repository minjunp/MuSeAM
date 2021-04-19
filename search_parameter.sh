#!/bin/bash

work_dir="/project/samee/minjun/museam_runs"
output_base_dir="${work_dir}/outputs"
par_output_dir="${work_dir}/pars"
fasta_file="/project/samee/minjun/methylation/classification/LACtrlF1_E13_combined.fa"
readout_file="/project/samee/minjun/methylation/classification/wt_readout.dat"
train_file="/project/samee/minjun/methylation/classification/train.py"

if [ -d ${output_base_dir} ]
then
	rm -rf ${output_base_dir}
fi
mkdir -p ${output_base_dir}

#exec_file="${work_dir}/parameter_search.sh"
#echo "#!/bin/bash" > ${exec_file}

#code="/project/samee/minjun/mpra/code/dummy.py"

index=0
for filters in 16 256 512
do
  for kernel_size in 12 16
  do
    for pool_type in Max
    do
      for regularizer in L_1
      do
        for activation_type in linear
        do
          for epochs in 30 40 50
          do
            for batch_size in 512
            do
              let index=index+1
              parameter_file_name="${par_output_dir}/parameters_"${index}".txt"
              echo "filters "${filters} > ${parameter_file_name}
              echo "kernel_size "${kernel_size} >> ${parameter_file_name}
              echo "pool_type "${pool_type} >> ${parameter_file_name}
              echo "regularizer "${regularizer} >> ${parameter_file_name}
              echo "activation_type "${activation_type} >> ${parameter_file_name}
              echo "epochs "${epochs} >> ${parameter_file_name}
              echo "batch_size "${batch_size} >> ${parameter_file_name}
              output_dir="${output_base_dir}/out_${index}"
              #if [ -d ${output_dir} ]
              #then
              #  rm -rf ${output_dir}
              #fi
              #mkdir -p ${output_dir}

          		python ${train_file} ${fasta_file} ${readout_file} ${parameter_file_name} > ${output_base_dir}/${index}.txt

              #echo "output_dir "${output_dir} >> ${parameter_file_name}
          		#echo "sleep 5m" >> ${exec_file}
            done
          done
        done
      done
    done
  done
done
#chmod +x ${exec_file}
