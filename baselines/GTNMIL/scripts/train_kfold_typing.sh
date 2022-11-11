export CUDA_VISIBLE_DEVICES=2
python main_kfold_typing.py \
--n_class 2 \
--data_path "data/gtnmil_data/graphs/ESCA_graphs_kimianet" \
--all_normal_set "data/gtnmil_data/graphs/ESCA_graphs/adeno_list_base.txt" \
--all_tumor_set "data/gtnmil_data/graphs/ESCA_graphs/squoumos_list_base.txt" \
--model_path "graph_transformer/saved_models/ESCA/" \
--log_path "graph_transformer/runs/ESCA/" \
--task_name "ESCA_typing" \
--batch_size 1 \
--train \
--log_interval_local 6 \
--num_fold 5 \
--test \