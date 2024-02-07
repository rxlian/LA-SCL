export CUDA_VISIBLE_DEVICES=1
epoch=2
max_len=128
bsz=32
steps=128

name=supcl
output_dir=result/dbpedia/300shot/$name
log_dir=result/dbpedia/300shot/$name/logs
mkdir "$output_dir"
mkdir "$log_dir"

python supcl2.py \
    --model_name_or_path bert-base-uncased \
    --tokenizer_name bert-base-uncased \
    --dataset DeveloperOats/DBPedia_Classes \
    --output_dir $output_dir \
    --num_train_epochs $epoch \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size $bsz \
    --learning_rate 1e-5 \
    --evaluation_steps $steps \
    --max_seq_length $max_len \
    --pad_to_max_length \
    --with_tracking \
    --checkpointing_steps $steps \
    --method supcl \
    --early_stop \
> >(tee -a $log_dir/stdout.log) \
2> >(tee -a $log_dir/stderr.log >&2)
