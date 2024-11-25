import re


def make_prompt(example: dict) -> str:
    '''
    Makes a zero-shot prompt for an MBPP example
    '''
    instruction = example['text']
    tests = "\n".join(example['test_list'])
    prompt = f'{instruction}\n\n```python\n{tests}\n```\n\nRespond with the function only wrapped in triple backticks.'
    return prompt


CODE_BLOCK_PATTERN = r"```(\w*)\n(.*?)\n```"

def extract_code(completion: str) -> str:
    match = re.findall(CODE_BLOCK_PATTERN, completion, flags=re.DOTALL)
    if match:
        # assume longest code block is desired code
        return max((m[1] for m in match), key=lambda x: len(x))
    else:
        return completion


FUNC_CALL_PATTERN = r"assert [A-z_]*\(.*?\) =="

def extract_func_calls(test_list: list[str]) -> list[str]:
    calls = []
    n_front, n_back = len("assert "), len(" ==")
    for test_str in test_list:
        match = re.findall(FUNC_CALL_PATTERN, test_str, flags=re.DOTALL)
        if match:
            call_str = match[0][n_front:-n_back]
            calls.append(call_str)
    return calls
            