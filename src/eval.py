#!/usr/bin/env python3
"""
多 GPU 并行评估脚本（保持单文件输出）
用法示例：
accelerate launch --num_processes 4 main.py \
    --data_file data.json \
    --model_name MiniCPM-V \
    --benchmark_name Streaming \
    --output_file result.json \
    --context_time 30
"""

import argparse
import json
import os
import tempfile
import shutil
from accelerate import Accelerator

# ----------------- 工具函数 -----------------
def load_data(path: str):
    """你的 utils.data_execution.load_data 实现"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ----------------- 主函数 -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",      required=True, help="原始数据文件")
    parser.add_argument("--model_name",     required=True, help="模型名称")
    parser.add_argument("--benchmark_name", required=True, help="Benchmark 名称")
    parser.add_argument("--output_file",    required=True, help="最终合并结果路径")
    parser.add_argument("--context_time",   type=int, required=True, help="上下文时长")
    args = parser.parse_args()

    accelerator = Accelerator()
    local_rank  = accelerator.local_process_index
    world_size  = accelerator.num_processes
    device      = accelerator.device

    # 1. 读全量数据
    full_data = load_data(args.data_file)

    # 2. 按 GPU 分片（round-robin）
    shard_data = full_data[local_rank::world_size]
    print(f"[Rank {local_rank}] 拿到 {len(shard_data)} 条数据")
    print(f"[StreamingBench] 实际处理 {len(full_data)} 条数据")
    # 3. 实例化模型
    model = None
    if args.model_name == "MiniCPM-V":
        from model.MiniCPMV import MiniCPMV
        model = MiniCPMV(device)
    elif args.model_name == "rekv":
        from model.rekv import rekv
        model = rekv(accelerator=accelerator)
    elif args.model_name == "MiniCPMo":
        from model.MiniCPMo import MiniCPMo
        model = MiniCPMo()
    else:
        raise ValueError(f"未知模型: {args.model_name}")
    model = accelerator.prepare(model)  # 让 accelerate 管理 device

    # 4. 实例化 benchmark
    benchmark = None
    if args.benchmark_name == "Streaming":
        from benchmark.StreamingBench import StreamingBench
        benchmark = StreamingBench(shard_data)
    elif args.benchmark_name == "StreamingProactive":
        from benchmark.StreamingBenchProactive import StreamingBenchProactive
        benchmark = StreamingBenchProactive(shard_data)
    elif args.benchmark_name == "StreamingSQA":
        from benchmark.StreamingBenchSQA import StreamingBenchSQA
        benchmark = StreamingBenchSQA(shard_data)
    elif args.benchmark_name == "StreamingOpenStreamText":
        from benchmark.StreamingOpenStreamText import StreamingOpenStreamText
        benchmark = StreamingOpenStreamText(shard_data)
    else:
        raise ValueError(f"未知 benchmark: {args.benchmark_name}")

    # 5. 创建临时目录保存各 rank 结果
    tmp_dir   = tempfile.mkdtemp(prefix="eval_")
    shard_out = os.path.join(tmp_dir, f"rank{local_rank}.json")

    # 6. 执行评估（每个 GPU 跑自己的 shard）
    benchmark.eval(shard_data, model, shard_out, args.context_time)

    # 7. 同步等待所有 rank 完成
    accelerator.wait_for_everyone()

    # 8. 主进程合并
    if accelerator.is_main_process:
        merged = []
        for fn in sorted(os.listdir(tmp_dir)):
            with open(os.path.join(tmp_dir, fn), 'r', encoding='utf-8') as f:
                merged.extend(json.load(f))

        # 按原始顺序重新排列（假设每条 question 有唯一 id）
        id2q = {q["id"]: q for d in merged for q in d.get("questions", [])}
        ordered = []
        for d in full_data:
            for q in d.get("questions", []):
                if q["id"] in id2q:
                    ordered.append(id2q[q["id"]])

        # 写回单文件
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False)

        # 清理临时目录
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()