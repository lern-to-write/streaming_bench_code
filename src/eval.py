#!/usr/bin/env python3
"""
多 GPU 并行评估脚本（保持单文件输出）
改进版：增加容错机制和更好的错误处理
"""

import argparse
import json
import os
import tempfile
import shutil
import time
import signal
import sys
import traceback
from accelerate import Accelerator

# ----------------- 工具函数 -----------------
def load_data(path: str):
    """你的 utils.data_execution.load_data 实现"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def timeout_handler(signum, frame):
    """超时处理函数"""
    raise TimeoutError("操作超时")

def safe_eval_with_timeout(benchmark, shard_data, model, shard_out, context_time, timeout_minutes=20):
    """带超时的安全评估函数"""
    try:
        # 设置超时信号
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)  # 设置超时时间
        
        # 执行评估
        benchmark.eval(shard_data, model, shard_out, context_time)
        
        # 取消超时信号
        signal.alarm(0)
        return True
        
    except TimeoutError:
        print(f"[WARNING] 评估超时 ({timeout_minutes}分钟)，尝试保存部分结果...")
        return False
    except Exception as e:
        print(f"[ERROR] 评估过程中发生错误: {e}")
        traceback.print_exc()
        return False
    finally:
        signal.alarm(0)  # 确保取消超时信号

# ----------------- 主函数 -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",      required=True, help="原始数据文件")
    parser.add_argument("--model_name",     required=True, help="模型名称")
    parser.add_argument("--benchmark_name", required=True, help="Benchmark 名称")
    parser.add_argument("--output_file",    required=True, help="最终合并结果路径")
    parser.add_argument("--context_time",   type=int, required=True, help="上下文时长")
    parser.add_argument("--timeout_minutes", type=int, default=30, help="单个rank最大处理时间（分钟）")
    parser.add_argument("--max_retries",    type=int, default=2, help="失败时最大重试次数")
    args = parser.parse_args()

    # 创建accelerator时增加超时设置
    accelerator = Accelerator()
    local_rank  = accelerator.local_process_index
    world_size  = accelerator.num_processes
    device      = accelerator.device

    print(f"[Rank {local_rank}] 开始处理，设备: {device}")

    try:
        # 1. 读全量数据
        full_data = load_data(args.data_file)

        # 2. 按 GPU 分片（round-robin）
        shard_data = full_data[local_rank::world_size]
        print(f"[Rank {local_rank}] 拿到 {len(shard_data)} 条数据")
        
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

        print(f"[Rank {local_rank}] 开始评估，输出到: {shard_out}")

        # 6. 执行评估（每个 GPU 跑自己的 shard）- 带重试机制
        success = False
        for retry in range(args.max_retries + 1):
            if retry > 0:
                print(f"[Rank {local_rank}] 第 {retry} 次重试...")
            
            success = safe_eval_with_timeout(
                benchmark, shard_data, model, shard_out, 
                args.context_time, args.timeout_minutes
            )
            
            if success:
                print(f"[Rank {local_rank}] 评估完成!")
                break
            else:
                if retry < args.max_retries:
                    print(f"[Rank {local_rank}] 评估失败，准备重试...")
                    time.sleep(5)  # 等待5秒后重试
                else:
                    print(f"[Rank {local_rank}] 评估最终失败，将创建空结果文件")
                    # 创建空结果文件以避免同步问题
                    with open(shard_out, 'w', encoding='utf-8') as f:
                        json.dump([], f)

        # 7. 同步等待所有 rank 完成 - 增加更长的超时时间
        print(f"[Rank {local_rank}] 等待其他进程完成...")
        
        # 使用更安全的同步方式
        try:
            # 设置环境变量增加超时时间
            os.environ["NCCL_BLOCKING_WAIT"] = "1"
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
            
            # 等待所有进程完成，增加更长的超时时间
            accelerator.wait_for_everyone()
            print(f"[Rank {local_rank}] 所有进程同步完成!")
            
        except Exception as e:
            print(f"[Rank {local_rank}] 同步等待时发生错误: {e}")
            # 即使同步失败，也继续处理，让主进程尝试合并现有结果
            pass

        # 8. 主进程合并结果
        if accelerator.is_main_process:
            print("[Main Process] 开始合并结果...")
            merged = []
            
            # 遍历临时目录中的所有结果文件
            result_files = [f for f in os.listdir(tmp_dir) if f.startswith("rank") and f.endswith(".json")]
            print(f"[Main Process] 找到 {len(result_files)} 个结果文件")
            
            for fn in sorted(result_files):
                file_path = os.path.join(tmp_dir, fn)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        merged.extend(data)
                    print(f"[Main Process] 成功加载 {fn}")
                except Exception as e:
                    print(f"[Main Process] 加载 {fn} 时出错: {e}")

            # 按原始顺序重新排列（假设每条 question 有唯一 id）
            if merged:
                try:
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
                    
                    print(f"[Main Process] 成功合并 {len(ordered)} 条结果到 {args.output_file}")
                except Exception as e:
                    print(f"[Main Process] 合并结果时出错: {e}")
                    # 如果合并失败，至少保存原始的merged数据
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        json.dump(merged, f, indent=2, ensure_ascii=False)
                    print(f"[Main Process] 保存了原始合并数据")
            else:
                print("[Main Process] 没有找到有效的结果数据")

            # 清理临时目录
            try:
                shutil.rmtree(tmp_dir)
                print("[Main Process] 清理临时目录完成")
            except Exception as e:
                print(f"[Main Process] 清理临时目录时出错: {e}")

    except Exception as e:
        print(f"[Rank {local_rank}] 主函数执行时发生严重错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()