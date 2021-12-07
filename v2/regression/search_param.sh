#!/bin/bash

work_dir="/Users/franciscogrisanti/Box/MuSeAM/regression"
output_base_dir="${work_dir}/outs/performance"
par_output_dir="${work_dir}/outs/pars"
true_pred_dir="${work_dir}/outs/true_pred"
fasta_file="${work_dir}/sequences.fa"
readout_file="${work_dir}/wt_readout.dat"
train_file="${work_dir}/train.py"

mkdir -p ${output_base_dir}
mkdir -p ${par_output_dir}
mkdir -p ${true_pred_dir}

index="francisco_scaling_0_1_linear_output"
filters=512
kernel_size=12
pool_type=Max
regularizer=L_1
activation_type=linear
epochs=30
batch_size=60 
loss_func=rank_mse
optimizer=adam
scaling=0_1

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
echo "scaling "${scaling} >> ${parameter_file_name}

python ${train_file} ${fasta_file} ${readout_file} ${parameter_file_name} > ${output_base_dir}/out_${index}.txt

mv true_vals.txt true_vals_${index}.txt
mv pred_vals.txt pred_vals_${index}.txt
mv true_vals_${index}.txt ${true_pred_dir}
mv pred_vals_${index}.txt ${true_pred_dir}
