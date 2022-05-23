# Extractive Question-Answering for the Malay Language

The development environment can be found in `requirements.txt`.

## Datasets

[Malay SQuAD](https://github.com/huseinzol05/malay-dataset/tree/master/question-answer/squad) released by [Malaya](https://github.com/huseinzol05/malaya).

## Finetune Models

We finetune the [bert-base-bahasa-cased](https://huggingface.co/malay-huggingface/bert-base-bahasa-cased/tree/main).

Run `bash run_train.sh` to fine-tune the model.

```
python3 train_malay.py --model_type bert \
        --model_name_or_path malay-huggingface/bert-base-bahasa-cased \
        --data_dir ./data_malay/ \
        --output_dir ~/saved/malay/bert-base\
        --tensorboard_save_path ./runs/malay-bert-base\
        --train_file ms-train-2.0.json \
        --predict_file ms-dev-2.0.json \
        --save_steps 1000 \
        --logging_steps 1000 \
        --begin_evaluation_steps 2500 \
        --do_train \
        --do_eval \
        --num_train_epochs 8\
        --evaluate_during_training \
        --learning_rate 2e-5 \
        --per_gpu_train_batch_size 48\
        --per_gpu_eval_batch_size 48\
        --gpu 7,6\
        --overwrite_output_dir \
        --version_2_with_negative \
        --max_seq_length 384 \
        --threads 10 \
```

## Evaluate Models

Run `bash run_eval.sh` to evaluated a saved model.
```
python3 eval_malay.py --model_type bert \
        --model_name_or_path ~/saved/malay/bert-base/checkpoint-best\
        --data_dir ./data_malay/ \
        --output_dir ./test/\
        --predict_file ms-dev-2.0.json \
        --do_eval \
        --per_gpu_eval_batch_size 32\
        --gpu 7,6\
        --version_2_with_negative \
        --max_seq_length 384 \
        --threads 10 \
```
