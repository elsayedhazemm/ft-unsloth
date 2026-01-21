from reward_funcs import accuracy_reward
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastModel
import torch
from qadataset import dataset

max_prompt_length = 256
max_completion_length = 2048

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it",
    max_seq_length=max_completion_length + max_prompt_length,  # Choose any for long context!
    full_finetuning=True,  # [NEW!] We have full finetuning now!
    fast_inference=True,
)

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Increase to 4 for smoother training
    num_generations=4,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    num_train_epochs = 2, # Set to 1 for a full training run
    save_steps=50,
    max_grad_norm=0.1,
    report_to="none",  # Can use Weights & Biases
    output_dir="outputs",
)


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        accuracy_reward
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[

    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()