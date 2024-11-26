import re
import matplotlib.pyplot as plt
import json
import pandas as pd
import torch

from abc import ABC, abstractmethod
from typing import List, Optional
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.distributions import Categorical

class BenchmarkDataset(ABC):
    def __init__(self, 
                 dataset: str, 
                 num_samples: int = 50, 
                 seed: int = 42, 
                 easy_mode: bool = True,  
                 num_proc: int = 8) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_samples = num_samples
        self.seed = seed
        self.easy_mode = easy_mode
        self.data = None
        self.system_prompt = ""
        self.num_proc = num_proc
        self.load_data()
        
    @abstractmethod
    def load_data(self):
        raise NotImplementedError
    
    def get_prompts(self, instruct: bool = False, tokenizer: Optional[AutoTokenizer] = None) -> List[str]:
        if not instruct:
            return [self.craft_prompt(example) for example in self.data]
        
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for instruction mode")
        
        messages = [
            [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': self.craft_prompt(example)}
            ]
            for example in self.data
        ]
        
        return tokenizer.apply_chat_template(messages, tokenize=False)
    
    @abstractmethod
    def craft_prompt(self, example: str) -> str:
        # few shot eval can be implemented here
        raise NotImplementedError
            
            
class FloresDataset(BenchmarkDataset):
    def __init__(self, 
                 dataset: str, 
                 num_samples: int = 50, 
                 seed: int = 42, 
                 easy_mode: bool = True,  
                 num_proc: int = 8) -> None:
        super().__init__(dataset, num_samples, seed, easy_mode, num_proc)
        self.system_prompt = 'You are a translation assistant. The user will provide you with a sentence in English and you will translate it to the indicated language.'
        self.target_lang = 'English' if self.easy_mode else 'Catalan'
    
    def load_data(self):
        self.data = load_dataset("openlanguagedata/flores_plus")['devtest']\
                    .filter(lambda x: x['iso_639_3'] == 'eng', num_proc=self.num_proc)\
                    .shuffle(seed=self.seed)\
                    .select(range(self.num_samples))['text']
                    
    def craft_prompt(self, example: str) -> str:
        return f"Translate the following sentence to {self.target_language}:\n{example}"
    
class GSMDataset(BenchmarkDataset):
    def __init__(self,
                 dataset: str, 
                 num_samples: int = 50, 
                 seed: int = 42, 
                 easy_mode: bool = True,  
                 num_proc: int = 8) -> None:
        self.filter_condition = (lambda x: x['answer'].count("<<") <= 3) if easy_mode \
                                else (lambda x: x['answer'].count("<<") > 3)
        super().__init__(dataset, num_samples, seed, easy_mode, num_proc)
        self.system_prompt = 'You are a math assistant. The user will ask you math questions and you will solve them.'
        
        
    
    def load_data(self):
        self.data = load_dataset('openai/gsm8k', 'main')['test']\
                    .filter(self.filter_condition, num_proc=self.num_proc)\
                    .shuffle(seed=self.seed)\
                    .select(range(self.num_samples))['question']
                    
    def craft_prompt(self, example: str) -> str:
        return example
        

@torch.no_grad()
def generate(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
    log_zero = -1e4

    # Initialize generated tokens with the input prompt
    generated_ids = input_ids
    finished_sequences = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=model.device)
    log_probs = []

    # Iteratively generate tokens using greedy decoding
    for token_idx in range(max_new_tokens):
        # Filter out finished sequences
        active_indices = torch.nonzero(~finished_sequences).squeeze(-1)
        if len(active_indices) == 0:
            break

        # Get model outputs for active sequences
        active_input_ids = generated_ids[active_indices]
        outputs = model(input_ids=active_input_ids)
        logits = outputs.logits

        # Get the last token logits and apply argmax to select the next token
        next_token_logits = logits[:, -1, :] / temperature
        next_token_log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        next_token_id = Categorical(logits=next_token_log_probs).sample()
        # next_token_log_prob, next_token_id = next_token_log_probs.max(dim=-1)

        # Save log next-token distribution for each sequence in batch; inactivate sequences produce <pad> token with probability 1
        curr_log_probs = torch.full((input_ids.shape[0], len(tokenizer)), log_zero, dtype=next_token_log_probs.dtype, device=model.device)
        curr_log_probs[:, tokenizer.pad_token_id] = 0.0
        curr_log_probs[active_indices] = next_token_log_probs
        log_probs.append(curr_log_probs)

        # Update finished sequences and add padding if necessary
        finished_sequences[active_indices] |= (next_token_id == tokenizer.eos_token_id)

        # Create a tensor for the next tokens to append to all sequences
        new_tokens = torch.full((generated_ids.shape[0], 1), tokenizer.pad_token_id, dtype=torch.long, device=model.device)
        new_tokens[active_indices] = next_token_id.unsqueeze(-1)

        # Append the next token to the generated sequence
        generated_ids = torch.cat([generated_ids, new_tokens], dim=-1)

    return generated_ids, log_probs

def update_vllm_config(vllm_config_path: str, vllm_log_file_path: str) -> dict:
    vllm_cfg_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": vllm_log_file_path,
                "mode": "a"
            }
        },
        "loggers": {
            "": {
                "level": "INFO",
                "handlers": ["file"]
            }
        }
    }
    with open(vllm_config_path, 'w') as f:
        json.dump(vllm_cfg_dict, f)

def parse_metrics(log_file_path) -> pd.DataFrame:
    """
    Parses the log file to extract the most recent Draft acceptance rate and System efficiency
    for each Number of speculative tokens setting, and visualizes them with separate plots.
    Additionally, calculates and plots the average SpecDecodeWorker stage times for all
    Number of speculative tokens settings.

    Args:
        log_file_path (str): The path to the log file.
    
    Returns:
        pd.DataFrame: A DataFrame containing the parsed metrics.
    """
    
    # Initialize data structures
    speculative_metrics = {}
    stage_times_dict = {}  # Key: Number of speculative tokens, Value: dict of times
    current_num_speculative_tokens = None

    # Regular expressions for parsing
    init_regex = re.compile(r"num_spec_tokens=(\d+)")
    metrics_regex = re.compile(
        r"Speculative metrics: Draft acceptance rate: ([\d\.]+), System efficiency: ([\d\.]+), Number of speculative tokens: (\d+)"
    )
    stage_times_regex = re.compile(
        r"average_time_per_proposal_tok_ms=([\d\.]+)\s+scoring_time_ms=([\d\.]+)\s+verification_time_ms=([\d\.]+)"
    )

    # Parse the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            line = line.strip()

            # Update current_num_speculative_tokens
            if 'Initializing an LLM engine' in line:
                init_match = init_regex.search(line)
                if init_match:
                    current_num_speculative_tokens = int(init_match.group(1))
                continue  # Move to the next line

            # Extract Speculative metrics
            metrics_match = metrics_regex.search(line)
            if metrics_match:
                draft_rate = float(metrics_match.group(1))
                system_eff = float(metrics_match.group(2))
                num_tokens = int(metrics_match.group(3))
                # Update with the most recent metrics
                speculative_metrics[num_tokens] = {
                    'draft_acceptance_rate': draft_rate,
                    'system_efficiency': system_eff
                }
                continue  # Move to the next line

            # Extract SpecDecodeWorker stage times
            stage_match = stage_times_regex.search(line)
            if stage_match and current_num_speculative_tokens is not None:
                avg_time = float(stage_match.group(1))
                scoring_time = float(stage_match.group(2))
                verification_time = float(stage_match.group(3))

                # Initialize the dictionary for current_num_speculative_tokens if not present
                if current_num_speculative_tokens not in stage_times_dict:
                    stage_times_dict[current_num_speculative_tokens] = {
                        'average_time_per_proposal_tok_ms': [],
                        'scoring_time_ms': [],
                        'verification_time_ms': []
                    }

                # Append the times
                stage_times_dict[current_num_speculative_tokens]['average_time_per_proposal_tok_ms'].append(avg_time)
                stage_times_dict[current_num_speculative_tokens]['scoring_time_ms'].append(scoring_time)
                stage_times_dict[current_num_speculative_tokens]['verification_time_ms'].append(verification_time)

    # Parse speculative_metrics
    sorted_tokens = sorted(speculative_metrics.keys())
    draft_rates = [speculative_metrics[num]['draft_acceptance_rate'] for num in sorted_tokens]
    system_efficiencies = [speculative_metrics[num]['system_efficiency'] for num in sorted_tokens]

    # Parse stage time metrics
    avg_time_per_proposal = []
    scoring_times = []
    verification_times = []
    tokens_with_stage_times = sorted(stage_times_dict.keys())

    for num_tokens in tokens_with_stage_times:
        times = stage_times_dict[num_tokens]
        avg_avg_time = sum(times['average_time_per_proposal_tok_ms']) / len(times['average_time_per_proposal_tok_ms'])
        avg_scoring_time = sum(times['scoring_time_ms']) / len(times['scoring_time_ms'])
        avg_verification_time = sum(times['verification_time_ms']) / len(times['verification_time_ms'])
        avg_time_per_proposal.append(avg_avg_time)
        scoring_times.append(avg_scoring_time)
        verification_times.append(avg_verification_time)

    # Prepare the DataFrame
    metrics = pd.DataFrame({
        'num_speculative_tokens': sorted_tokens,
        'draft_acceptance_rate': draft_rates,
        'system_efficiency': system_efficiencies,
        'average_time_per_proposal_tok_ms': avg_time_per_proposal,
        'scoring_time_ms': scoring_times,
        'verification_time_ms': verification_times
    })
    
    return metrics

def plot_figure(x, y1, y2, xlabel, ylabel, title, label1, label2, save_folder=None):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, marker='o', linestyle='-', color='blue', label=label1)
    plt.plot(x, y2, marker='o', linestyle='-', color='red', label=label2)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True)
    plt.legend(title="Task", fontsize=14, title_fontsize=16)
    if save_folder:
        plt.savefig(f"{save_folder}/{title}.png")
    else:
        plt.show()
    
def plot_bar_chart(x, y1, y2, xlabel, ylabel, title, label1, label2, save_folder=None):
    fig, ax = plt.subplots()
    bar_width = 0.35
    rects1 = ax.bar(x - bar_width/2, y1, bar_width, label=label1, color='blue')
    rects2 = ax.bar(x + bar_width/2, y2, bar_width, label=label2, color='red')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend(title="Task", fontsize=14, title_fontsize=16)
    if save_folder:
        plt.savefig(f"{save_folder}/{title}.png")
    else:
        plt.show()
    
def plot_metrics(
    metrics_control: pd.DataFrame, 
    metrics_experiment: pd.DataFrame, 
    control_label: str, 
    experiment_label: str,
    save_folder: Optional[str] = None
):
    """
    Plots the Draft acceptance rate, System efficiency, Average Time per Proposal Token, Scoring Time,
    and Verification Time for the control and experiment settings.

    Args:
        metrics_control (pd.DataFrame): DataFrame containing the metrics for the control setting.
        metrics_experiment (pd.DataFrame): DataFrame containing the metrics for the experiment setting.
        control_label (str): Label for the control setting.
        experiment_label (str): Label for the experiment setting.
    """
    # Plot Draft Acceptance Rate
    plot_figure(
        metrics_control['num_speculative_tokens'], 
        metrics_control['draft_acceptance_rate'], 
        metrics_experiment['draft_acceptance_rate'], 
        'Number of Speculative Tokens', 
        'Draft Acceptance Rate', 
        'Draft Acceptance Rate vs Number of Speculative Tokens', 
        control_label, 
        experiment_label,
        save_folder
    )

    # Plot System Efficiency
    plot_figure(
        metrics_control['num_speculative_tokens'], 
        metrics_control['system_efficiency'], 
        metrics_experiment['system_efficiency'], 
        'Number of Speculative Tokens', 
        'System Efficiency', 
        'System Efficiency vs Number of Speculative Tokens', 
        control_label, 
        experiment_label,
        save_folder
    )

    # Plot Average Time per Proposal Token
    plot_figure(
        metrics_control['num_speculative_tokens'], 
        metrics_control['average_time_per_proposal_tok_ms'], 
        metrics_experiment['average_time_per_proposal_tok_ms'], 
        'Number of Speculative Tokens', 
        'Average Time per Proposal Token (ms)', 
        'Average Time per Proposal Token vs Number of Speculative Tokens', 
        control_label, 
        experiment_label,
        save_folder
    )
    
    # Plot Scoring Time
    plot_figure(
        metrics_control['num_speculative_tokens'], 
        metrics_control['scoring_time_ms'], 
        metrics_experiment['scoring_time_ms'], 
        'Number of Speculative Tokens', 
        'Scoring Time (ms)', 
        'Scoring Time vs Number of Speculative Tokens', 
        control_label, 
        experiment_label,
        save_folder
    )
    
    # Plot Verification Time
    plot_figure(
        metrics_control['num_speculative_tokens'], 
        metrics_control['verification_time_ms'], 
        metrics_experiment['verification_time_ms'], 
        'Number of Speculative Tokens', 
        'Verification Time (ms)', 
        'Verification Time vs Number of Speculative Tokens', 
        control_label, 
        experiment_label,
        save_folder
    )
    
    # Plot Runtime vs Number of Speculative Tokens
    plot_bar_chart(
        metrics_control['num_speculative_tokens'], 
        metrics_control['vllm_latencies'], 
        metrics_experiment['vllm_latencies'], 
        'Number of Speculative Tokens',
        'Runtime (s)',
        'Runtime vs Number of Speculative Tokens',
        control_label,
        experiment_label,
        save_folder
    )

def print_cpu_gpu_times(trace_file_path: str):
    """
    Parses a PyTorch autograd profiler trace file and prints the total CPU and GPU times.

    Args:
        trace_file_path (str): Path to the JSON trace file.
    """
    try:
        with open(trace_file_path, 'r') as f:
            trace = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{trace_file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: File '{trace_file_path}' is not a valid JSON file.")
        return

    events = trace.get('traceEvents', [])
    if not events:
        print("No trace events found in the file.")
        return

    total_cpu_time = 0.0
    total_gpu_time = 0.0
    total_other_time = 0.0

    for event in events:
        if event.get('ph') == 'X':  # Complete events
            duration = event.get('dur', 0.0)
            category = event.get('cat', '')

            if 'cpu' in category:
                total_cpu_time += duration
            elif 'gpu' in category or 'cuda' in category:
                total_gpu_time += duration
            else:
                print(f"Unknown category: {category}")
                total_other_time += duration

    # Convert durations from microseconds to milliseconds
    total_cpu_time_ms = total_cpu_time
    total_gpu_time_ms = total_gpu_time
    total_other_time_ms = total_other_time

    print(f"Total CPU Time: {total_cpu_time_ms:.3f} ms")
    print(f"Total GPU Time: {total_gpu_time_ms:.3f} ms")
    print(f"Total Other Time: {total_other_time_ms:.3f} ms")
    