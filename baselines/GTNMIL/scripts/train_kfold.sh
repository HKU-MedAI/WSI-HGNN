export CUDA_VISIBLE_DEVICES=1
python main_kfold.py \
--n_class 2 \
--data_path "data/gtnmil_data/graphs/COAD_graphs_kimianet" \
--all_normal_set "data/gtnmil_data/graphs/COAD_graphs/normal_list_base.txt" \
--all_tumor_set "data/gtnmil_data/graphs/COAD_graphs/tumor_list_base.txt" \
--model_path "graph_transformer/saved_models/COAD/" \
--log_path "graph_transformer/runs/COAD/" \
--task_name "COAD_cls_kimianet_v2" \
--batch_size 8 \
--train \
--log_interval_local 6 \
--num_fold 5 \
--test \