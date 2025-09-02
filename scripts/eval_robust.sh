#!/bin/bash

cd /home/u_2359629761/yy_code/StreamingBench/src

# 设置更稳定的环境变量
export NCCL_TIMEOUT=1800  # 30分钟超时
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0

# 设置模型和参数
EVAL_MODEL="rekv"
CONTEXT_TIME=-1
TASK="real"
DATA_FILE="./data/questions_${TASK}.json"
OUTPUT_FILE="./data/${TASK}_output_${EVAL_MODEL}.json"
BENCHMARK="Streaming"

echo "开始执行分布式评估..."
echo "模型: $EVAL_MODEL"
echo "任务: $TASK"
echo "数据文件: $DATA_FILE"
echo "输出文件: $OUTPUT_FILE"

# 使用改进的启动命令
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch \
    --num_processes 8 \
    eval.py \
    --model_name "$EVAL_MODEL" \
    --benchmark_name "$BENCHMARK" \
    --data_file "$DATA_FILE" \
    --output_file "$OUTPUT_FILE" \
    --context_time $CONTEXT_TIME \
    --timeout_minutes 30 \
    --max_retries 4

echo "评估完成!"
