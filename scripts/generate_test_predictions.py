import os
import json
import time
import torch

os.environ["HF_HOME"] = r"E:\hf_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = r"E:\ai-job-search\applications\vima3ya\Assesment\submission\adapter"
TEST_INPUTS_PATH = r"E:\ai-job-search\applications\vima3ya\Assesment\training_assessment\candidate_pack\data\test_inputs.jsonl"
OUTPUT_PREDS_PATH = r"E:\ai-job-search\applications\vima3ya\Assesment\submission\predictions_test.jsonl"

def main():
    print("=" * 60)
    print("Starting Test Set Inference on Qwen2.5-3B-Instruct + QLoRA")
    print("Base Model:", BASE_MODEL_ID)
    print("Adapter Path:", ADAPTER_PATH)
    print("HF_HOME Cache:", os.environ["HF_HOME"])
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading base model in 4-bit on GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("Model loaded successfully!")

    print(f"Reading test inputs from: {TEST_INPUTS_PATH}")
    test_rows = []
    with open(TEST_INPUTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_rows.append(json.loads(line))
    print(f"Loaded {len(test_rows)} test cases.")

    predictions = []
    start_time = time.time()

    for idx, row in enumerate(test_rows, 1):
        test_id = row["id"]
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        out_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        predictions.append({"id": test_id, "output": out_text})

        if idx % 20 == 0 or idx == len(test_rows):
            elapsed = time.time() - start_time
            rate = idx / elapsed
            rem = (len(test_rows) - idx) / rate if rate > 0 else 0
            print(f"[{idx:3d}/{len(test_rows)}] ({rate:.2f} it/s, ETA {rem:.1f}s) ID: {test_id} | Sample: {out_text[:60]}...")

    print(f"\nWriting predictions to {OUTPUT_PREDS_PATH}...")
    with open(OUTPUT_PREDS_PATH, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n--- Running Validation on predictions_test.jsonl ---")
    with open(OUTPUT_PREDS_PATH, "r", encoding="utf-8") as f:
        read_lines = [json.loads(l) for l in f]

    print(f"Total lines: {len(read_lines)}")
    assert len(read_lines) == len(test_rows), f"Expected {len(test_rows)}, got {len(read_lines)}"

    REQUIRED_KEYS = {"category", "sub_category", "priority", "sentiment", "urgency", "needs_human", "txn_id", "txn_date"}
    valid_json_count = 0
    all_keys_count = 0

    for i, p in enumerate(read_lines):
        assert p["id"] == test_rows[i]["id"], f"ID mismatch at line {i}: {p['id']} vs {test_rows[i]['id']}"
        try:
            parsed = json.loads(p["output"])
            valid_json_count += 1
            if isinstance(parsed, dict) and set(parsed.keys()) == REQUIRED_KEYS:
                all_keys_count += 1
        except Exception:
            pass

    print(f"Valid JSON syntax: {valid_json_count}/{len(read_lines)} ({valid_json_count / len(read_lines) * 100:.2f}%)")
    print(f"Exact 8-key schema match: {all_keys_count}/{len(read_lines)} ({all_keys_count / len(read_lines) * 100:.2f}%)")
    print("SUCCESS: predictions_test.jsonl generated and verified!")

if __name__ == "__main__":
    main()
