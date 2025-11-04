import unsloth
from unsloth import FastLanguageModel
import torch

fourbit_models = [
    "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", # Qwen 14B 2x faster
    "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
    "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",

    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/Phi-4",
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    max_seq_length = 2048,   # Choose any for long context!
    load_in_4bit = False,    # 4 bit quantization to reduce memory
    load_in_8bit = False,    # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

import torch
import gc

# Clear cache and collect garbage
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    qat_scheme = "int4",
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

for module in model.modules():
    if "FakeQuantized" in module.__class__.__name__:
        print("QAT is applied!")
        break
    
from datasets import load_from_disk
dataset = load_from_disk('musicpile_cluster_filtered_v2')

def to_qwen_format(example):
    text = example["text"]
    if "</s> Assistant:" in text:
        instruction, response = text.split("</s> Assistant:", 1)
        example["text"] = (
            f"<|im_start|>user\n{instruction.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n{response.strip()}<|im_end|>"
        )
    return example

dataset = dataset.map(to_qwen_format)

dataset = dataset.shuffle(seed=3407)
split = int(0.9 * len(dataset))
trainset = dataset.select(range(split))
testset  = dataset.select(range(split, len(dataset)))

print(dataset)
print(trainset)
print(testset)
# print(trainset[0])

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = trainset,
    eval_dataset = testset, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
        ddp_find_unused_parameters = False
    ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

# Clear cache and collect garbage
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

trainer_stats = trainer.train()

from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig

quantize_(model, QATConfig(step = "convert"))

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")