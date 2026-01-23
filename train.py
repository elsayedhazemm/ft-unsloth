from reward_funcs import accuracy_reward, match_format_exactly, match_format_approximately
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastModel
import torch
from qadataset import dataset
import os

os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_PROJECT"] = ""
os.environ["WANDB_ENTITY"] = ""
os.environ["HF_TOKEN"] = ""

max_prompt_length = 256
max_completion_length = 4096

model, tokenizer = FastModel.from_pretrained(
    model_name="google/medgemma-1.5-4b-it",
    max_seq_length=max_completion_length + max_prompt_length,  # Choose any for long context!
    full_finetuning=True,  # [NEW!] We have full finetuning now!
    fast_inference=True,
    token=""
)

training_args = GRPOConfig(
    learning_rate=1e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.01,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Increase to 4 for smoother training
    num_generations=8,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_train_epochs = 3, # Set to 1 for a full training run
    save_steps=50,
    max_grad_norm=0.1,
    report_to="wandb",  # Can use Weights & Biases
    output_dir="outputs",
    use_vllm=True
)


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        accuracy_reward,           # 0.0 to 3.0 - correctness is the main signal
        match_format_exactly,      # 0.0 to 1.0 - bonus for perfect format
        match_format_approximately # -1.0 to 1.0 - partial credit for format elements
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()