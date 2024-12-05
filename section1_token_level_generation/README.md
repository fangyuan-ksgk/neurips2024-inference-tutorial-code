# Section 1: Token-level generation

In this section, we present example useage of **token-level** generation techniques using the `guidance` library.

## File Structure
Code for this section is structured as follows:
- `Speculative_Decoding_Demo.ipynb` -- a Jupyter Notebook with a minimal implementation of Speculative Decoding. This version is **not** optimized and is intended only for educational purposes. It closely matches the code present in the tutorial slides.
- `speculative_decoding_vllm_benchmark.py` provides an example usage of Speculative Decoding with vLLM, a highly optimized library for efficient generation.
- `speculative_decoding_utils.py` contains helper logic for data manipulation / autoregressive generation.

## Speculative Decoding Benchmark
Using `speculative_decoding_vllm_benchmark.py`, we capture vLLM Speculative Decoding metrics such as: 
- Draft Acceptance Rate: The empirical acceptance rate of the proposal method on a per-token basis. This is useful for evaluating how well the proposal method aligns with the scoring method.
- System Efficiency: The empirical efficiency, measured as the number of tokens emitted by the system divided by the number of tokens that could be emitted by the system if the proposal method were perfect.
- Average Time per Proposal Token (ms): Time spent proposing the draft tokens divided by number of draft tokens.
- Scoring Time (ms): Total time spent scoring the draft tokens.
- Verification Time (ms): Total time spent verifying the draft tokens.

Example command line:
```bash
python speculative_decoding_vllm_benchmark.py
    --target_model_name meta-llama/Meta-Llama-3.1-70B-Instruct
    --helper_model_name meta-llama/Meta-Llama-3.2-1B-Instruct
    --instruct_mode 
    --dataset gsm 
    --num_samples 512
    --seed 42 
    --max_new_tokens 512
    --max_speculation_size 20
    --num_proc 6 
    --temperature 1.0
```


### Benchmark Results
Here are some example results we get from running the benchmark on an H100 80GB GPU. Note that the right column represents the "misaligned" case when draft/helper model coupled with the instruct model. In this case the acceptance rates are substantially lower.

| Llama 3.1 70B Instruct <> Llama 3.2 1B Instruct |  Llama 3.1 70B Instruct <> Llama 3.2 1B |
| - | - |
| ![Acceptance Rate](<vllm_benchmark_h100/Draft Acceptance Rate vs Number of Speculative Tokens.png>) | ![Acceptance Rate](<vllm_benchmark_h100_wbase/Draft Acceptance Rate vs Number of Speculative Tokens.png>) |
| ![Efficiency](<vllm_benchmark_h100/System Efficiency vs Number of Speculative Tokens.png>) | ![Efficiency](<vllm_benchmark_h100_wbase/System Efficiency vs Number of Speculative Tokens.png>) |
| ![Time Per Proposal](<vllm_benchmark_h100/Average Time per Proposal Token vs Number of Speculative Tokens.png>) | ![Time Per Proposal](<vllm_benchmark_h100_wbase/Average Time per Proposal Token vs Number of Speculative Tokens.png>) |
| ![Scoring Time](<vllm_benchmark_h100/Scoring Time vs Number of Speculative Tokens.png>) | ![Scoring Time](<vllm_benchmark_h100_wbase/Scoring Time vs Number of Speculative Tokens.png>) |
| ![Verification Time](<vllm_benchmark_h100/Verification Time vs Number of Speculative Tokens.png>) | ![Verification Time](<vllm_benchmark_h100_wbase/Verification Time vs Number of Speculative Tokens.png>) |
| ![Average Latency](<vllm_benchmark_h100/Runtime vs Number of Speculative Tokens.png>) | ![Average Latency](<vllm_benchmark_h100_wbase/Runtime vs Number of Speculative Tokens.png>) |
