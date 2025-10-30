# score-based metrics
python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_multi_config.json" --model-name "LLM.ASTERModel" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 1, "seq_len": 4, "lr": 0.001, "llm_dim": 768, "num_epochs": 100}' --adapter "llm_adapter" --gpus 1 --num-workers 2 --timeout 60000 --save-path "score/ASTERModelfull" --data-name-list "PSM.csv" --seed 2021

# label-based metrics
python ./scripts/run_benchmark.py --config-path "unfixed_detect_label_multi_config.json" --model-name "LLM.ASTERModel" --model-hyper-params '{"horizon": 1, "is_train": 1, "norm": true, "sampling_rate": 1, "seq_len": 4, "lr": 0.001, "llm_dim": 768, "num_epochs": 100}' --adapter "llm_adapter" --gpus 1 --num-workers 2 --timeout 60000 --save-path "label/ASTERModelfull" --data-name-list "PSM.csv" --seed 2021
