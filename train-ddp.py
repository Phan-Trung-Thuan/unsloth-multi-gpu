#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA fine-tuning for Qwen3-0.6B on MusicPile dataset with Unsloth + DDP
======================================================================
Usage:
  torchrun --nproc_per_node=2 train_sft.py

Notes:
- Each process is pinned to a single GPU via LOCAL_RANK.
- Uses LoRA + 4bit quantization + gradient checkpointing.
- Dataset is loaded from HuggingFace dataset saved via `save_to_disk`.
"""

import os
import torch
import logging
from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import shutil

# --------------------------- DDP bootstrap ---------------------------
LOCAL_RANK  = int(os.environ.get("LOCAL_RANK", 0))
RANK        = int(os.environ.get("RANK", 0))
WORLD_SIZE  = int(os.environ.get("WORLD_SIZE", 1))
IS_DIST     = WORLD_SIZE > 1

if IS_DIST:
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(LOCAL_RANK)
device = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO if RANK == 0 else logging.WARNING)
log = logging.getLogger("train_sft")
log.info(f"RANK={RANK} LOCAL_RANK={LOCAL_RANK} WORLD_SIZE={WORLD_SIZE} device={device}")

# --------------------------- User config -----------------------------
LORA_RANK      = 16
NUM_EPOCHS     = 1
MODEL_PATH     = "Qwen/Qwen3-8B"
MAX_LEN        = 1024
LR             = 2e-4
SAVE_STEPS     = 1000
TARGET_GLOBAL_BATCH = 2
RESUME_FROM    = None  # e.g., "checkpoints/checkpoint-1000"

DATASET_PATH   = "musicpile_cluster_filtered"  # your saved dataset folder

# Effective batch size calculation
per_device_bs = max(1, TARGET_GLOBAL_BATCH // max(1, WORLD_SIZE))
grad_accum    = 1
log.info(f"per_device_train_batch_size={per_device_bs}, gradient_accumulation_steps={grad_accum}")

# ---------------------- 1) Load model/tokenizer ----------------------
device_map = {"": f"cuda:{LOCAL_RANK}"} if torch.cuda.is_available() else None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name        = MODEL_PATH,
    max_seq_length    = MAX_LEN,
    load_in_4bit      = False,
    load_in_8bit      = False,
    full_finetuning   = False,
    device_map        = device_map,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_RANK,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = LORA_RANK,
    lora_dropout               = 0.0,
    bias                       = "none",
    qat_scheme = "int4", # Quantized Awareness Training
    use_gradient_checkpointing = "unsloth",
    random_state               = 3407,
)

for module in model.modules():
    if "FakeQuantized" in module.__class__.__name__:
        log.info("QAT is applied!")
        break

# ----------------------- 2) Load your dataset -----------------------
log.info("Loading dataset from disk...")
dataset = load_from_disk(DATASET_PATH)

# Shuffle and split
dataset = dataset.shuffle(seed=3407)
split = int(0.9 * len(dataset))
trainset = dataset.select(range(split))
testset  = dataset.select(range(split, len(dataset)))

log.info(f"Dataset loaded: {len(trainset)} train / {len(testset)} test samples")

# ---------------------- 3) SFT config -------------------------------
out_dir = f"./checkpoints_musicpile_qwen3_0.6B_lora{LORA_RANK}"

cfg = SFTConfig(
    dataset_text_field           = "text",
    per_device_train_batch_size  = per_device_bs,
    gradient_accumulation_steps  = grad_accum,
    num_train_epochs             = NUM_EPOCHS,
    learning_rate                = LR,
    warmup_ratio                 = 0.03,
    lr_scheduler_type            = "linear",
    max_grad_norm                = 1.0,
    weight_decay                 = 0.01,
    seed                         = 3407,
    output_dir                   = out_dir,
    save_strategy                = "epoch",
    # save_steps                   = SAVE_STEPS,
    save_total_limit             = 20,
    logging_steps                = 1,
    report_to                    = "none",
    optim                        = "adamw_8bit",
    ddp_find_unused_parameters   = False,
    eval_strategy                = 'epoch',
    metric_for_best_model        = 'eval_loss',
    greater_is_better            = False,
    load_best_model_at_end       = True,
)

# -------------------- 4) Gradient Guard Callback ---------------------
class GradientGuard(TrainerCallback):
    """Skip a step if grad_norm explodes."""
    def on_step_end(self, args, state, control, **kwargs):
        gn = kwargs.get("logs", {}).get("grad_norm")
        if gn is None and state.log_history:
            gn = state.log_history[-1].get("grad_norm")
        if gn is not None and gn > 100:
            if RANK == 0:
                print(f"[Guard] grad_norm={gn:.1f} -> skip step {state.global_step}")
            control.should_skip_next_step = True

# ----------------------- 5) Train -----------------------------------
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = trainset,
    eval_dataset  = testset,
    args          = cfg,
    callbacks     = [GradientGuard()],
)

if RANK == 0:
    log.info("🚀 Start training on main process...")
train_stats = trainer.train(resume_from_checkpoint=RESUME_FROM)

# ----------------------- 6) Save final model ------------------------
if RANK == 0:
    save_dir = f"./lora_musicpile_qwen3_8B_rank{LORA_RANK}_epoch{NUM_EPOCHS}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Training complete. Saved to: {save_dir}")

    if os.path.isdir(save_dir): # Ensure the directory exists before zipping
        print(f"Starting compression of {save_dir}...")
        
        archive_base_name = save_dir 
        
        zip_path = shutil.make_archive(
            base_name=archive_base_name, 
            format='zip', 
            root_dir='.',      # Start the search from the current directory
            base_dir=save_dir  # Archive the folder itself
        )
        
        print(f"✅ Model zipped successfully! File is: {zip_path}")
        print("You can now download this single .zip file from Vast.ai.")

# ----------------------- 7) Cleanup ---------------------------------
if IS_DIST and torch.distributed.is_initialized():
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
