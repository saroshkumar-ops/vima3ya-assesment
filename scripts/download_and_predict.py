import os
import shutil
import urllib.request
import time
import json
import torch

INCOMPLETE_FILE = r"E:\hf_models\Qwen2.5-3B-Instruct\.cache\huggingface\download\aoe4E07IMh7reFyUkVoVk040mQk=.67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2.3baaba05.incomplete"
TARGET_FILE = r"E:\hf_models\Qwen2.5-3B-Instruct\model-00001-of-00002.safetensors"
TOTAL_BYTES = 3968658944
URL = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/model-00001-of-00002.safetensors"

BASE_MODEL_DIR = r"E:\hf_models\Qwen2.5-3B-Instruct"
ADAPTER_PATH = r"E:\ai-job-search\applications\vima3ya\Assesment\submission\adapter"
TEST_INPUTS_PATH = r"E:\ai-job-search\applications\vima3ya\Assesment\training_assessment\candidate_pack\data\test_inputs.jsonl"
OUTPUT_PREDS_PATH = r"E:\ai-job-search\applications\vima3ya\Assesment\submission\predictions_test.jsonl"

def step1_finish_download():
    print("=" * 60)
    print("STEP 1: Resuming download of model-00001-of-00002.safetensors")
    print("=" * 60)

    if not os.path.exists(TARGET_FILE):
        if os.path.exists(INCOMPLETE_FILE):
            print(f"Copying {INCOMPLETE_FILE} to {TARGET_FILE}...")
            shutil.copyfile(INCOMPLETE_FILE, TARGET_FILE)
        else:
            with open(TARGET_FILE, "wb") as f:
                pass

    current_size = os.path.getsize(TARGET_FILE)
    print(f"Current file size: {current_size:,} bytes ({current_size / (1024**2):.1f} MB / {TOTAL_BYTES / (1024**2):.1f} MB)")

    if current_size >= TOTAL_BYTES:
        print("model-00001-of-00002.safetensors is already completely downloaded!")
        return

    req = urllib.request.Request(
        URL,
        headers={
            "Range": f"bytes={current_size}-{TOTAL_BYTES - 1}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
    )

    start_time = time.time()
    last_print = start_time
    bytes_downloaded = 0
    chunk_size = 4 * 1024 * 1024 # 4MB chunks

    print(f"Connecting to Hugging Face CDN (offset {current_size:,})...")
    with urllib.request.urlopen(req, timeout=30) as resp:
        with open(TARGET_FILE, "ab") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bytes_downloaded += len(chunk)
                now = time.time()
                if now - last_print > 3:
                    elapsed = now - start_time
                    speed = bytes_downloaded / (elapsed * 1024 * 1024) if elapsed > 0 else 0
                    total_now = current_size + bytes_downloaded
                    pct = total_now / TOTAL_BYTES * 100
                    rem_mb = (TOTAL_BYTES - total_now) / (1024 * 1024)
                    eta_sec = rem_mb / speed if speed > 0 else 0
                    print(f"[{pct:5.1f}%] {total_now / (1024**2):.1f} / {TOTAL_BYTES / (1024**2):.1f} MB | {speed:.2f} MB/s | ETA: {eta_sec:.0f}s")
                    last_print = now

    final_size = os.path.getsize(TARGET_FILE)
    print(f"Download complete! Final size: {final_size:,} bytes")
    assert final_size == TOTAL_BYTES, f"Expected {TOTAL_BYTES}, got {final_size}"

def step2_generate_predictions():
    print("\n" + "=" * 60)
    print("STEP 2: Running Test Set Inference on GPU")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer from local directory...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading 4-bit quantized base model onto RTX 3050 GPU...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"Loading winning LoRA adapter from: {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("Model and adapter successfully initialized!")

    print(f"Reading test inputs from: {TEST_INPUTS_PATH}...")
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

        if idx % 25 == 0 or idx == len(test_rows):
            elapsed = time.time() - start_time
            rate = idx / elapsed
            rem = (len(test_rows) - idx) / rate if rate > 0 else 0
            print(f"[{idx:3d}/{len(test_rows)}] ({rate:.2f} it/s, ETA {rem:.1f}s) ID: {test_id} | Sample: {out_text[:60]}...")

    print(f"\nWriting test predictions to {OUTPUT_PREDS_PATH}...")
    with open(OUTPUT_PREDS_PATH, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n--- Validating predictions_test.jsonl ---")
    with open(OUTPUT_PREDS_PATH, "r", encoding="utf-8") as f:
        read_lines = [json.loads(l) for l in f]

    assert len(read_lines) == 400, f"Expected 400 lines, got {len(read_lines)}"
    valid_json = sum(1 for p in read_lines if is_valid_json(p["output"]))
    print(f"Total lines: {len(read_lines)}")
    print(f"Valid JSON syntax: {valid_json}/400 ({valid_json/4:.1f}%)")
    print("ALL CHECKS PASSED: predictions_test.jsonl is ready!")

def is_valid_json(s):
    try:
        json.loads(s)
        return True
    except:
        return False

if __name__ == "__main__":
    step1_finish_download()
    step2_generate_predictions()
