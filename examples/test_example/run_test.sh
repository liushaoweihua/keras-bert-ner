DATA_DIR="../data"
MODEL_DIR="../models"
OUTPUT_DIR="../test_outputs"

python run_test.py \
    -test_data ${DATA_DIR}/test.txt \
    -model_configs ${MODEL_DIR}/model_configs.json \
    -output_path ${OUTPUT_DIR} \
    -device_map "0"