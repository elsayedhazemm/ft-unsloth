import unsloth
from reward_funcs import accuracy_reward, match_format_exactly, match_format_approximately
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
import torch
from qadataset import dataset
import os



max_prompt_length = 256
max_completion_length = 2048
max_seq_length = max_completion_length + max_prompt_length
lora_rank = 16
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
      gpu_memory_utilization = 0.8, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
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
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=2,  # Decrease if out of memory
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