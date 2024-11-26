import torch
import pandas as pd
import gc
import os
import time

from argparse import ArgumentParser
from matplotlib import pyplot as plt
from time import sleep
from typing import Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
import multiprocessing as mp

from speculative_decoding_utils import (
    parse_metrics,
    plot_metrics,
    FloresDataset, 
    GSMDataset, 
    update_vllm_config
)

datasets = {
    "flores": FloresDataset,
    "gsm": GSMDataset
}

def config_vllm(easy_mode: bool, args) -> Tuple[str, str]:
    clean_model_name = args.target_model_name.replace("/", "_")
    vllm_log_file = f"vllm_{clean_model_name}_{args.dataset}_{'easy' if easy_mode else 'hard'}_spec_{args.speculation_size}.log"
    vllm_logging_config_path = f"vllm_{clean_model_name}_{args.dataset}_{'easy' if easy_mode else 'hard'}_spec_{args.speculation_size}.json"
    update_vllm_config(vllm_logging_config_path, vllm_log_file)
    os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
    os.environ["VLLM_LOGGING_CONFIG_PATH"] = vllm_logging_config_path
    return vllm_log_file, vllm_logging_config_path
    

def run_benchmark(easy_mode:bool, args, conn) -> pd.DataFrame:
    vllm_log_file, vllm_logging_config_path = config_vllm(easy_mode, args)
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
    dataset = datasets[args.dataset](
        args.dataset, 
        args.num_samples, 
        args.seed, 
        easy_mode, 
        args.num_proc
    )
    prompts = dataset.get_prompts(instruct=args.instruct_mode, tokenizer=tokenizer)

    # remove <begin_of_text> since vllm's tokenizer with add that too.
    prompts = [p.replace(tokenizer.bos_token,"") for p in prompts]
    
    sampling_params = SamplingParams(temperature=args.temperature, seed=args.seed, max_tokens=args.max_new_tokens)
    vllm_args = {
        "model": args.target_model_name,
        "distributed_executor_backend": "mp",
        "tensor_parallel_size": torch.cuda.device_count(),
        "max_model_len": 4096,
        "speculative_model": args.helper_model_name,
        "use_v2_block_manager": True,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.6,
        "dtype": torch.bfloat16
    }
    
    if args.quantized:
        vllm_args["quantization"] = "bitsandbytes"
        vllm_args["load_format"] = "bitsandbytes"
    
    mean_vllm_latencies = []
    # for specualtion_size in range(1, args.max_speculation_size + 1):
    print(f"Dataset: {args.dataset}, Easy Mode: {easy_mode}, Speculation Size: {args.speculation_size}")
    llm = LLM(
        **vllm_args,
        num_speculative_tokens=args.speculation_size,
        disable_log_stats=False
    )

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    latency = time.perf_counter() - start
    mean_vllm_latencies.append(latency/len(prompts))
    
    metrics = parse_metrics(vllm_log_file)
    metrics["vllm_latencies"] = mean_vllm_latencies
    
    # return metrics
    conn.send(metrics)
    conn.close()
    

def main():
    parser = ArgumentParser()
    parser.add_argument("--target_model_name", type=str, default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
    parser.add_argument("--helper_model_name", type=str, default="unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--instruct_mode", action="store_true")
    parser.add_argument("--dataset", type=str, default="gsm", choices=["flores", "gsm"])
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_speculation_size", type=int, default=5)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    
    save_folder = "./vllm_benchmark_h100_wbase"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    easy_metrics = []
    for spec_size in range(1, args.max_speculation_size+1):
        args.speculation_size = spec_size
        parent_conn, child_conn = mp.Pipe()
        easy_process = mp.Process(target=run_benchmark, args=(True, args, child_conn))
        easy_process.start()
        easy_process.join()
        easy_metrics.append(parent_conn.recv())
    easy_metrics = pd.concat(easy_metrics)
    easy_metrics.to_csv(f"{save_folder}/easy_metrics_{args.dataset}.csv")
    
    hard_metrics = []
    for spec_size in range(1, args.max_speculation_size+1):
        args.speculation_size = spec_size
        parent_conn, child_conn = mp.Pipe()
        hard_process = mp.Process(target=run_benchmark, args=(False, args, child_conn))
        hard_process.start()
        hard_process.join()
        hard_metrics.append(parent_conn.recv())
    hard_metrics = pd.concat(hard_metrics)
    hard_metrics.to_csv(f"{save_folder}/hard_metrics_{args.dataset}.csv")    
    
    plot_metrics(
        metrics_control=easy_metrics, 
        metrics_experiment=hard_metrics, 
        control_label=f"{args.dataset} - easy",
        experiment_label=f"{args.dataset} - hard",
        save_folder=save_folder
    )
    
if __name__ == "__main__":
    main()