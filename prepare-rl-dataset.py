from datasets import load_from_disk, Dataset
from tqdm import tqdm

# --- Setup ---
ds = load_from_disk('musicpile_cluster_filtered')
ds = ds.remove_columns('embedding')
print(ds)

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import gc
import torch
from datasets import load_from_disk  # if loading existing ds

# --- 1. Split text into prompt and chosen ---
def split_text(example):
    if " </s> Assistant:" in example["text"]:
        parts = example["text"].split(" </s> Assistant:")
        example["prompt"] = parts[0].replace("Human: ", "").strip()
        example["chosen"] = parts[1].strip()
    else:
        example["prompt"] = example["text"]
        example["chosen"] = ""
    return example

# Example: assume `ds` is already loaded
# ds = load_from_disk("your_dataset_path")

ds = ds.map(split_text)
print("✅ Added prompt and chosen columns:", ds.column_names)

# --- 2. Load Qwen3-0.6B with vLLM ---
model_id = "./Qwen3-1.7B"

# tokenizer = AutoTokenizer.from_pretrained(model_id)

# vLLM uses its own optimized engine (multi-GPU aware)
llm = LLM(
    model=model_id,
    # tokenizer=tokenizer,
    tensor_parallel_size=torch.cuda.device_count(),  # use all GPUs
    dtype="bfloat16",  # or "float16" if needed
    gpu_memory_utilization=0.95, # utilize 90% of VRAM,
    trust_remote_code=True,
    enforce_eager=False,
    max_num_seqs=512
)
print('Finished llm object')

# --- 3. Sampling configuration ---
sampling_params = SamplingParams(
    temperature=1.2,
    top_p=0.9,
    max_tokens=512,
    repetition_penalty=1.0,
    n=1,  # one sample per prompt
)
print('Finished sampling_params object')

new_rejected = []
batch_size = 256
num_rows = len(ds)

for start in tqdm(range(0, len(ds), batch_size)):
    batch = ds[start:start + batch_size]
    prompts = [f"Human: {p}\nAssistant:" for p in batch["prompt"]]

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    new_rejected.extend([o.outputs[0].text.strip() for o in outputs])
    # break

# --- 5. Add to dataset and save ---
# ds = ds.select(range(batch_size))
ds = ds.add_column("rejected", new_rejected)

save_dir = "musicpile_rlhf_pairs_vllm"
ds.save_to_disk(save_dir)
print(f"✅ Saved dataset to: {save_dir}")

import shutil
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