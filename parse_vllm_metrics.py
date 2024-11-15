import re
import matplotlib.pyplot as plt
import json

def parse_and_plot_metrics(log_file_path):
    """
    Parses the log file to extract the most recent Draft acceptance rate and System efficiency
    for each Number of speculative tokens setting, and visualizes them with separate plots.
    Additionally, calculates and plots the average SpecDecodeWorker stage times for all
    Number of speculative tokens settings.

    Args:
        log_file_path (str): The path to the log file.
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

    # Prepare data for plotting
    sorted_tokens = sorted(speculative_metrics.keys())
    draft_rates = [speculative_metrics[num]['draft_acceptance_rate'] for num in sorted_tokens]
    system_efficiencies = [speculative_metrics[num]['system_efficiency'] for num in sorted_tokens]

    # Plot Draft Acceptance Rate vs Number of Speculative Tokens
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_tokens, draft_rates, marker='o', linestyle='-', color='blue')
    plt.title('Draft Acceptance Rate vs Number of Speculative Tokens')
    plt.xlabel('Number of Speculative Tokens')
    plt.ylabel('Draft Acceptance Rate')
    plt.grid(True)
    plt.xticks(sorted_tokens)  # Ensure all token numbers are shown on x-axis
    plt.show()

    # Plot System Efficiency vs Number of Speculative Tokens
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_tokens, system_efficiencies, marker='s', linestyle='--', color='green')
    plt.title('System Efficiency vs Number of Speculative Tokens')
    plt.xlabel('Number of Speculative Tokens')
    plt.ylabel('System Efficiency')
    plt.grid(True)
    plt.xticks(sorted_tokens)
    plt.show()

    # Calculate and Plot Average Stage Times for all Number of Speculative Tokens
    if stage_times_dict:
        # Initialize lists for plotting
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

        # Plot Average Time per Proposal Token
        plt.figure(figsize=(10, 6))
        plt.plot(tokens_with_stage_times, avg_time_per_proposal, marker='o', linestyle='-', color='purple')
        plt.title('Average Time per Proposal Token vs Number of Speculative Tokens')
        plt.xlabel('Number of Speculative Tokens')
        plt.ylabel('Average Time per Proposal Token (ms)')
        plt.grid(True)
        plt.xticks(tokens_with_stage_times)
        plt.show()

        # Plot Scoring Time
        plt.figure(figsize=(10, 6))
        plt.plot(tokens_with_stage_times, scoring_times, marker='s', linestyle='--', color='orange')
        plt.title('Scoring Time vs Number of Speculative Tokens')
        plt.xlabel('Number of Speculative Tokens')
        plt.ylabel('Scoring Time (ms)')
        plt.grid(True)
        plt.xticks(tokens_with_stage_times)
        plt.show()

        # Plot Verification Time
        plt.figure(figsize=(10, 6))
        plt.plot(tokens_with_stage_times, verification_times, marker='^', linestyle='-.', color='red')
        plt.title('Verification Time vs Number of Speculative Tokens')
        plt.xlabel('Number of Speculative Tokens')
        plt.ylabel('Verification Time (ms)')
        plt.grid(True)
        plt.xticks(tokens_with_stage_times)
        plt.show()
    else:
        # If there are no stage times recorded
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No SpecDecodeWorker stage times data available.',
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=14, color='red')
        plt.axis('off')
        plt.show()




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
    