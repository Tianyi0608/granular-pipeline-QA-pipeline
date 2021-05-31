export CUDA_VISIBLE_DEVICES=2

# the ref bp json file we need to run QA on
export BP_JSON_REF_FILE_PATH='../data/bp_json/granular.eng-provided-72.0pct.devtest-15.0pct.ref.d.bp.json'
# the directory for all the data inputs and outputs in this pipeline model
export DATA_DIR='./data'
# the system bp json file used to run the scorer, created in step 4
export BP_JSON_SYS_FILE_PATH='bp_json_sys.bp.json'
# the path of score.py
export SCORER_PATH='/shared/better/granular/bp_lib-v1.3.6.1'

# pip install -U pip # update pip
# pip install pytokenizations

# step 1: read bp json datafile and convert to qnli binary classification csv format as well as squad json format
# binary classification: as the binary classification model is pre-trained on wnli, we need three labels: 0 for no-answer, 1 for nothing, and 2 for has-answer
# argument extraction: the squad json file incorporates all the questions for both has-answer and no-answer
# the model only predicts all as has-answer
# the prediction for no-answer will be deleted during post processing.
# at this step, doc_template.json, csv_input.csv, csv_answer.csv, squad_input.json will be generated under $DATA_DIR
echo "start pre-processing converting to binary classification csv..."
python process_data.py \
    --pre_processing \
    --bp_json_ref_file_path $BP_JSON_REF_FILE_PATH \
    --data_dir $DATA_DIR
echo "end pre-processing"

# step 2: csv as input to run the binary classification model
# pay attention to the path
echo "start running binary classification"
python run_glue.py \
    --model_name_or_path ../models/binary_squad2_better_roberta_large \
    --cache_dir ../cache \
    --train_file ../data/template_qa/train_binary_wnli.csv \
    --validation_file ../data/template_qa/dev_binary_wnli.csv \
    --test_file $DATA_DIR/csv_input.csv \
    --do_predict \
    --max_seq_length 300 \
    --output_dir $DATA_DIR \
    --overwrite_output_dir
echo "end binary classification"

# step 3: squad json as input to run the argument extraction model
echo "start running argument extraction"
python run_squad.py \
    --model_type roberta \
    --model_name_or_path ../models/roberta-large_has_answer_string/best_65.91 \
    --cache_dir ../cache \
    --overwrite_cache \
    --do_eval \
    --do_lower_case \
    --n_best_size 15 \
    --max_answer_length 15 \
    --max_seq_length 300 \
    --doc_stride 100 \
    --max_query_length 15 \
    --output_dir $DATA_DIR \
    --predict_file $DATA_DIR/squad_input.json \
    --per_gpu_eval_batch_size=64
echo "end argument extraction"

# step 4: convert the output to bp json format
echo 'start post-processing...'
python process_data.py \
    --post_processing \
    --data_dir $DATA_DIR \
    --bp_json_ref_file_path $BP_JSON_REF_FILE_PATH \
    --bp_json_sys_file_path $DATA_DIR/$BP_JSON_SYS_FILE_PATH
echo 'end post-processing'

# step 5: run better scorer
echo 'start running scorer...'
python $SCORER_PATH/score.py -v Granular \
$DATA_DIR/$BP_JSON_SYS_FILE_PATH $BP_JSON_REF_FILE_PATH