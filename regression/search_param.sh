#!/bin/bash

work_dir="/project/samee/minjun/MuSeAM/regression"
output_base_dir="${work_dir}/ou/performance"
par_output_dir="${work_dir}/outs/pars"
true_pred_dir="${work_dir}/outs/true_pred"
fasta_file="${work_dir}/sequences.fa"
readout_file="${work_dir}/wt_readout.dat"
train_file="${work_dir}/train.py"

#if [ -d ${output_base_dir} ]
#then
#	rm -rf ${output_base_dir}
#fi
mkdir -p ${output_base_dir}
mkdir -p ${par_output_dir}
mkdir -p ${true_pred_dir}

#exec_file="${work_dir}/parameter_search.sh"
#echo "#!/bin/bash" > ${exec_file}

#code="/project/samee/minjun/mpra/code/dummy.py"

index=0
for filters in 512
do
  for kernel_size in 12
  do
    for pool_type in Max
    do
      for regularizer in L_1
      do
        for activation_type in linear
        do
          for epochs in 30 40 50 60
          do
            for batch_size in 60
            do
              for loss_func in mse rank_mse
              do
                for optimizer in adam
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
                  echo "loss_func "${loss_func} >> ${parameter_file_name}
                  echo "optimizer "${optimizer} >> ${parameter_file_name}
                  output_dir="${output_base_dir}/out_${index}"

                  python ${train_file} ${fasta_file} ${readout_file} ${parameter_file_name} > ${output_base_dir}/${index}.txt

                  mv true_vals.txt true_vals_${index}.txt
                  mv pred_vals.txt pred_vals_${index}.txt
                  mv true_vals_${index}.txt ${true_pred_dir}
                  mv pred_vals_${index}.txt ${true_pred_dir}

                  #echo "output_dir "${output_dir} >> ${parameter_file_name}
              		#echo "sleep 5m" >> ${exec_file}
                done
              done
            done
          done
        done
      done
    done
  done
done
#chmod +x ${exec_file}
