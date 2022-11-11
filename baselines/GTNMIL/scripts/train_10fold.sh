export CUDA_VISIBLE_DEVICES=3
python main_10fold.py \
--n_class 2 \
--data_path "data/gtnmil_data/graphs/COAD_graphs" \
--all_normal_set "data/gtnmil_data/graphs/COAD_graphs/normal_list_base.txt" \
--all_tumor_set "data/gtnmil_data/graphs/COAD_graphs/tumor_list_base.txt" \
--model_path "graph_transformer/saved_models/COAD/" \
--log_path "graph_transformer/runs/COAD/" \
--task_name "COAD_cls_10fold" \
--batch_size 8 \
--train \
--log_interval_local 6 \
--num_fold 10 \
--test \