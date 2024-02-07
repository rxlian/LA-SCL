export CUDA_VISIBLE_DEVICES=1

epoch=10
name=scaled_ii
echo $name
# output_dir=result/dbpedia/lp/fewshot/$name
# log_dir=result/dbpedia/lp/fewshot/$name/logs
output_dir=''

model_dir2=result/newsgroups20/baseline/$name

# mkdir "$output_dir"
# mkdir "$log_dir"

python3 supcl.py \
	--model_name_or_path bert-base-uncased \
	--train_file data/newsgroups20/train.csv \
	--valid_file data/newsgroups20/valid.csv \
	--dataset SetFit/20_newsgroups \
	--max_seq_length 128 \
	--output_dir $output_dir/ \
	--learning_rate 5e-3 \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--num_train_epochs $epoch \
	--weight_decay 0.01 \
	--overwrite_output_dir True \
	--load_best_model_at_end True \
	--evaluation_strategy epoch \
	--save_strategy epoch \
	--metric_for_best_model eval_accuracy \
	--logging_steps 50 \
	--save_total_limit 1 \
	--task direct_test \
> >(tee -a $log_dir/stdout.log) \
2> >(tee -a $log_dir/stderr.log >&2)
rm -rf $output_dir/checkpoint*/
