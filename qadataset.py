from datasets import load_dataset

# Load your local file (assumes it is a standard JSON list of objects)
raw_dataset = load_dataset("json", data_files="data/rl-a.json", split="train")


def map_grpo_format(example):
    # Extract the user's question from the first turn
    user_prompt = example["conversations"][0]["value"]
    full_resp = example["conversations"][1]["value"]

    return {
        "prompt": [
            {"role": "user", "content": user_prompt}
        ],
        "answer": full_resp
    }
    
# Remove old columns to keep the dataset clean
dataset = raw_dataset.map(map_grpo_format, remove_columns=raw_dataset.column_names)

# Verify the first row
print(dataset[0])