data_types=(and or xor)
act_types=(relu sigmoid tanh)

for data_type in ${data_types[@]}; do
	for act_type in ${act_types[@]}; do
		echo ====================================================================
		echo data_type: ${data_type}, act_type: ${act_type}
		echo ====================================================================
		python train.py --data_type ${data_type} --act_type ${act_type}
		python train.py --data_type ${data_type} --act_type ${act_type} --use_bn		
	done
done