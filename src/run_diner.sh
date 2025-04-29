ARTS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Counterfactual=1
GPU=0
fusion_mode=sum
dataset_name=$1
# dataset_name in ["laptop", "rest"]
seed=20
epoch=40
batch_size=60
weight_decay=0.01
learning_rate=5e-5
max_len_s=120
max_len_a=13
model_name=XLMr
checkpoint=1
test_model=1
test_model_path=""

if [ $Counterfactual = 1 ];then
    if [ $ARTS = 1 ];then   
        save_dir="output/${dataset_name}_ARTS/data_${dataset_name}_ARTS-lr_${learning_rate}-bz_${batch_size}_${fusion_mode}_${model_name}"
    else
        save_dir="output/${dataset_name}/data_${dataset_name}-lr_${learning_rate}-bz_${batch_size}_${fusion_mode}_${model_name}_${model_name}"
    fi
else
    if [ $ARTS = 1 ];then
        save_dir="output/results/${dataset_name}_ARTS/data_${dataset_name}_ARTS-lr_${learning_rate}-bz_${batch_size}_${model_name}"
    else
        save_dir="output/results/${dataset_name}/data_${dataset_name}-lr_${learning_rate}-bz_${batch_size}_${model_name}"
    fi
fi
mkdir -p $save_dir

python3 main_cfabsa.py \
    --ARTS $ARTS \
    --Counterfactual $Counterfactual\
    --GPU $GPU\
    --fusion_mode $fusion_mode\
    --dataset_name $dataset_name \
    --seed $seed \
    --epoch $epoch \
    --batch_size $batch_size \
    --weight_decay $weight_decay \
    --learning_rate $learning_rate \
    --max_len_s $max_len_s \
    --max_len_a $max_len_a \
    --model_name $model_name \
    --checkpoint $checkpoint \
    --test_model $test_model \
    --test_model_path $test_model_path \
    --save_dir $save_dir 2>&1 | tee  ${save_dir}/run.log
