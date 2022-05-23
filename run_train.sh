#! /bin/sh
for time in {1..1}
do
	python3 train_malay.py --model_type bert \
	       	--model_name_or_path malay-huggingface/bert-base-bahasa-cased \
	       	--data_dir ./data_malay/ \
			--output_dir ~/saved/malay/bert-base\
			--tensorboard_save_path ./runs/malay-bert-base\
			--train_file ms-train-2.0.json \
			--predict_file ms-dev-2.0.json \
			--save_steps 500 \
			--logging_steps 500 \
			--begin_evaluation_steps 2500 \
			--do_train \
			--do_eval \
			--num_train_epochs 8\
			--evaluate_during_training \
			--learning_rate 2e-5 \
			--per_gpu_train_batch_size 24\
			--per_gpu_eval_batch_size 24\
			--gpu 7,6\
			--overwrite_output_dir \
			--version_2_with_negative \
			--max_seq_length 384 \
            --threads 50 \
        
done
