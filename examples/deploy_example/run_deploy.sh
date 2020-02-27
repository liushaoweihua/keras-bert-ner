MODEL_DIR="../models"

python run_deploy.py \
    -model_configs ${MODEL_DIR}/model_configs.json \
    -log_path ${MODEL_DIR}/deploy_log/ \
    -device_map "cpu"