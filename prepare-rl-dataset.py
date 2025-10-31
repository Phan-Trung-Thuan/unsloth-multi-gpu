#!/usr/bin/env python3
def main():
    from datasets import load_from_disk
    from tqdm import tqdm
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Setup ---
    ds = load_from_disk('musicpile_cluster_filtered')
    ds = ds.remove_columns('embedding')
    print(ds)

    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"

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

    ds = ds.map(split_text)
    print("✅ Added prompt and chosen columns:", ds.column_names)

    # --- 2. Load Qwen3-0.6B normally (not vLLM) ---
    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # automatically spread across GPUs if available
        trust_remote_code=True
    )
    model.eval()
    print("✅ Loaded model and tokenizer")

    # --- 3. Generation config ---
    generation_kwargs = {
        "temperature": 1.2,
        "top_p": 0.9,
        "max_new_tokens": 512,
        "repetition_penalty": 1.0,
        "do_sample": True
    }

    new_rejected = []
    batch_size = 2

    # --- 4. Inference loop ---
    for start in tqdm(range(0, len(ds), batch_size)):
        batch = ds[start:start + batch_size]
        prompts = [f"Human: {p}\nAssistant:" for p in batch["prompt"]]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Extract only the assistant part
        cleaned = []
        for full_text, prompt in zip(decoded, prompts):
            if "Assistant:" in full_text:
                cleaned.append(full_text.split("Assistant:")[-1].strip())
            else:
                cleaned.append(full_text.strip())
        new_rejected.extend(cleaned)

    # --- 5. Add to dataset and save ---
    ds = ds.add_column("rejected", new_rejected)

    save_dir = "musicpile_rlhf_pairs_hf"
    ds.save_to_disk(save_dir)
    print(f"✅ Saved dataset to: {save_dir}")

    # --- 6. Optional compression ---
    import shutil
    if os.path.isdir(save_dir):
        print(f"Starting compression of {save_dir}...")
        zip_path = shutil.make_archive(base_name=save_dir, format="zip", root_dir=".", base_dir=save_dir)
        print(f"✅ Zipped dataset: {zip_path}")
        print("You can now download this .zip file from Vast.ai.")

if __name__ == "__main__":
    main()
