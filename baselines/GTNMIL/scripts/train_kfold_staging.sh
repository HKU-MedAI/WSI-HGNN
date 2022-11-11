export CUDA_VISIBLE_DEVICES=0
python main_kfold_staging.py \
--n_class 4 \
--data_path "data/gtnmil_data/graphs/BRCA_graphs_v2" \
--all_stage_0_set "data/gtnmil_data/graphs/BRCA_graphs_v2/stage_0_list_base.txt" \
--all_stage_1_set "data/gtnmil_data/graphs/BRCA_graphs_v2/stage_1_list_base.txt" \
--all_stage_2_set "data/gtnmil_data/graphs/BRCA_graphs_v2/stage_2_list_base.txt" \
--all_stage_3_set "data/gtnmil_data/graphs/BRCA_graphs_v2/stage_3_list_base.txt" \
--model_path "graph_transformer/saved_models/BRCA/" \
--log_path "graph_transformer/runs/BRCA/" \
--task_name "BRCA_staging" \
--batch_size 8 \
--train \
--log_interval_local 6 \
--num_fold 5 \
--test \
