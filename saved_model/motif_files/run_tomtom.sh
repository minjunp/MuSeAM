#!/bin/bash

database_file="$(pwd)/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme"
pwms="$(pwd)/pwms"

tomtom_output="$(pwd)/tomtom_outputs"
if [ -d ${tomtom_output} ]
then
	rm -rf ${tomtom_output}
fi
mkdir -p ${tomtom_output}

for f in `ls ${pwms}`
do
	n=`echo $f | cut -d"_" -f2 | cut -d"." -f1`
	#echo $n
	pwm_file="${pwms}/pwm_${n}.mat"
	output_dir="${tomtom_output}/result_${n}"
	#mkdir -p ${output_dir}
	#tomtom [options] <query file> <target file>+
	tomtom -oc ${output_dir} -evalue -thresh 5 ${pwm_file} ${database_file}
done
