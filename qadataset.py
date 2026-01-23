from datasets import load_dataset

# Load your local file (assumes it is a standard JSON list of objects)
raw_dataset = load_dataset("json", data_files="data/rl-a.json", split="train")


SYSTEM_PROMPT = """\
You are a medical expert. Think through the problem carefully, then provide your answer.
Place your reasoning between <think> and </think>.
Then provide your final answer between <answer> and </answer>.

Example format:
<think>
[Your step-by-step reasoning here]
</think>
<answer>
[Your final answer here]
</answer>
"""

def map_grpo_format(example):
    # Extract the user's question from the first turn
    user_prompt = example["conversations"][0]["value"]
    full_resp = example["conversations"][1]["value"]

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "answer": full_resp
    }
    
# Remove old columns to keep the dataset clean
dataset = raw_dataset.map(map_grpo_format, remove_columns=raw_dataset.column_names)

# Verify the first row
print(dataset[0])