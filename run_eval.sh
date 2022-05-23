#! /bin/sh
for time in {1..1}
do
	python3 eval_malay.py --model_type bert \
	       	--model_name_or_path ~/saved/malay/bert-base/checkpoint-best\
	       	--data_dir ./data_malay/ \
			--output_dir ./test/\
			--predict_file ms-dev-2.0.json \
			--do_eval \
			--per_gpu_eval_batch_size 32\
			--gpu 7,6,5,4,3,2,1,0\
			--version_2_with_negative \
			--max_seq_length 384 \
            --threads 10 \
        
done
