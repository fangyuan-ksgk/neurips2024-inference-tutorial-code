# Section 2: Meta-generation

## MBPP Case Study

### Files
- [mbpp_demo.ipynb](./mbpp/mbpp_demo.ipynb) contains example implementations of three primitve meta-generation algorithms: best-of-$n$, minimum Bayes risk, and refinement. The algorithms (and variants thereof) are then compared on a Python code generation task with the Mostly Basic Python Problems (MBPP) dataset.
- [mbpp_utils.py](./mbpp/mbpp_utils.py) contains helpers for executing and processing Python programs.

### Setup

While the code is set up to use an Open AI API key or an Open AI compatible endpoint, it uses the `litellm` library for model inference, which supports a variety of inference providers and model backends; see [here](https://arxiv.org/abs/2304.05128) for a full list. As such, the code in the notebook should be compatible with other backends beside Open AI, modulo some [differences in parameter names](https://docs.litellm.ai/docs/completion/input). 

To use the code with OpenAI, just set `OPENAI_API_KEY` in your environment.
```bash
export OPENAI_API_KEY=<YOUR-API-KEY>
```


## Treefinement

### Files
- [treefinement.ipynb notebook](./treefinement/treefinement.ipynb) shows a more complex meta-generation algorithm called **Treefinement** [[Aggarwal et al 2024]()] that combines parallel sampling, refinement, and tree search. The task is formally verified code generation: generating Rust code that passes a formal verifier called Verus. 
- [utils.py](./treefinement/utils.py) contains utilities for executing and processing Rust code and helpers for tree search.

### Setup (Verus)

To run the notebook, you will need the Verus verifier. Please follow the [Installation Instructions](https://github.com/verus-lang/verus/blob/main/INSTALL.md) from the Verus repository. The tutorial was developed and tested on Mac OS, using commit `50d07b5fe4465fed8b76f4d050c945ba5dd17141` of [Verus](https://github.com/verus-lang/verus).

After following the instructions, you will have a path to the verifier (in the instructions, it is `./target-verus/release/verus`). Set the `VERUS_PATH` environment to this path:
```bash
export VERUS_PATH=/my/path/to/target-verus/release/verus
```



### Setup (LLM)

By default, the notebook requires an Open AI API key. After obtaining a key, set the `OPENAI_API_KEY` environment variable to the key:
```bash
export OPENAI_API_KEY=<YOUR-API-KEY>
```

In general, a language model that is hosted with an Open AI API compatible endpoint can be used. You will need to pass in your API's endpoint URL (and API key, if applicable) when instantiating `openai.Client()` in the notebook.

#### Run the notebook!
After following these steps, please proceed with the notebook.
