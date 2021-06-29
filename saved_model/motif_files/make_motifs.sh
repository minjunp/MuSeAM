#!/bin/bash

src_dir="../"
header="${src_dir}/motif_files/header"

pwms="${src_dir}/motif_files/pwms"
if [ -d ${pwms} ]
then
	rm -rf ${pwms}
fi
mkdir -p ${pwms}

logos="${src_dir}/motif_files/logos"
if [ -d ${logos} ]
then
	rm -rf ${logos}
fi
mkdir -p ${logos}

for f in `ls ${src_dir}/*.txt`
do
	n=`echo $f | rev | cut -d"_" -f1 | rev | cut -d"." -f1`
	#echo $n
	pwm_file="${pwms}/pwm_${n}.mat"
	head -9 ${header} > ${pwm_file}
	echo "MOTIF motif_${n}" >> ${pwm_file}
	tail -1 ${header} >> ${pwm_file}
	cat $f >> ${pwm_file}

	#ceqlogo -i meme.motifs -m MA0036.1 -o logo.eps
	fwd_logo_file="${logos}/fwd_${n}.png"
	ceqlogo -i ${pwm_file} -m motif_${n} -f PNG -o ${fwd_logo_file}

	rc_logo_file="${logos}/rc_${n}.png"
	ceqlogo -i ${pwm_file} -m motif_${n} -r -f PNG -o ${rc_logo_file}
done
