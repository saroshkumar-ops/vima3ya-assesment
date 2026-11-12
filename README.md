# NimbusPay Ticket Triage — LLM Fine-Tuning Submission (VIMA3YA)

This repository contains the end-to-end solution for the VIMA3YA Machine Learning Intern (LLM Fine-Tuning) assessment: fine-tuning a small instruction-tuned model (`<= 4B` parameters) with PEFT LoRA/QLoRA to map raw NimbusPay support tickets into strict 8-field JSON objects matching `SCHEMA.md`.

## Repository Contents

- [`notebook.ipynb`](notebook.ipynb) — End-to-end Google Colab T4 notebook covering Part A (data audit & cleaning), Part B (dev baselines), Part C (chat-template formatting, `-100` assistant-only masking, evidence-based `max_length`), Part D (5 single-variable controlled experiments), Part E (error analysis), and final export + reproducibility verification.
- [`cleaning_log.csv`](cleaning_log.csv) — Audit log of all `355` dropped (`180`) and repaired (`175`) rows from `train.jsonl` (`1,445 -> 1,265` clean rows).
- [`scripts/audit_and_clean.py`](scripts/audit_and_clean.py) — Standalone reproducible CLI script implementing Part A audit & cleaning.
- [`scripts/build_notebook.py`](scripts/build_notebook.py) — Reproducible generator script for `notebook.ipynb`.
- `adapter/` — *[Populated after Colab T4 run]* Final PEFT LoRA adapter (`adapter_config.json` + `adapter_model.safetensors`, `< 100 MB`).
- `predictions_test.jsonl` — *[Populated after Colab T4 run]* Raw greedy-decoded (`max_new_tokens=200`, whitespace `.strip()` only) outputs for all `400` rows in `test_inputs.jsonl`.
- `report.md` — *[Finalized after Colab T4 run]* Concise technical report (`<= 3` pages).

> Note: Raw assessment data files (`train.jsonl`, `dev.jsonl`, `test_inputs.jsonl`) and intermediate `work_data/` / checkpoints are excluded via `.gitignore` per submission rules.

## Approach Summary

1. **Base Model & Tooling:** `Qwen/Qwen2.5-3B-Instruct` (`~3.09B` parameters), Hugging Face `transformers`, `peft`, `trl` (`SFTTrainer`), and `bitsandbytes` (4-bit NF4 QLoRA and 16-bit `fp16` LoRA comparison).
2. **Part A Data Audit (`1,445 -> 1,265` rows, `355` defects):**
   - **Dropped (`180` rows):** `30` exact `dev.jsonl` leaks, `15` perturbed near-exact `dev.jsonl` leaks, `60` exact in-train duplicates, `40` perturbed near-duplicates, `25` truncated/invalid JSON labels, and `10` junk/empty placeholder tickets.
   - **Repaired (`175` rows, `205` field-level fixes):** `70` legacy v1 `ivr-transcript` rows using `"prio"` (`low`/`medium`/`high` -> `"priority"` `P3`/`P2`/`P1`, of which `30` also needed `DD/MM/YYYY` -> `YYYY-MM-DD` `txn_date` fixes), `60` v2 `priority` rule violations, `25` `amount_paise` values recorded in rupees (`x100`), and `20` unnormalized `txn_id` strings.
3. **Part C Tokenization & Masking:**
   - Verified `tokenizer.pad_token` (`<|endoftext|>`, ID `151643`) != `tokenizer.eos_token` (`<|im_end|>`, ID `151645`).
   - Applied `-100` loss masking to all prompt tokens (`system` + `user` + assistant generation header) so cross-entropy trains solely on the target JSON + `<|im_end|>`.
   - Measured token lengths across `train_clean` (`max=2,215`, `p97=268`, `36` rows `>512`), `dev` (`max=2,181`, `3` rows `>512`), and `test_prompt` (`max=2,089`, `11` rows `>512`); selected `max_length=2560` (`0` truncated rows) and included `max_length=512` as an ablation.
4. **Part B & Part D Experiments (Colab T4 16 GB):**
   - Baseline 1 (zero-shot fixed prompt) & Baseline 2 (rules-in-prompt reference)
   - `exp1_raw_qlora`: Raw `train.jsonl` (1,445 rows), 4-bit QLoRA `r=16, alpha=32, lr=2e-4, epochs=3, max_length=2560`
   - `exp2_clean_qlora` (Anchor): Cleaned `train_clean.jsonl` (1,265 rows), 4-bit QLoRA `r=16, alpha=32, lr=2e-4, epochs=3, max_length=2560`
   - `exp3_clean_fp16_lora`: 16-bit `fp16` LoRA vs 4-bit QLoRA
   - `exp4_clean_qlora_r8`: 4-bit QLoRA `r=8, alpha=16` vs `r=16, alpha=32`
   - `exp5_clean_qlora_len512`: 4-bit QLoRA `max_length=512` vs `max_length=2560`

## Results

*[Pending execution of `notebook.ipynb` on Google Colab T4.]*

## How to Reproduce on Google Colab (Free T4 GPU)

1. Upload `candidate_pack/` (or `training_assessment.zip`) to Google Drive at `MyDrive/vima3ya_assessment/candidate_pack` (or `/content/candidate_pack`).
2. Open [`notebook.ipynb`](notebook.ipynb) in Google Colab and set **Runtime -> Change runtime type -> T4 GPU**.
3. Run all cells top-to-bottom (`Runtime -> Run all`). Completed stages automatically checkpoint under `MyDrive/vima3ya_assessment/outputs/` (`SKIP_COMPLETED = True`).
4. Collect final outputs from `MyDrive/vima3ya_assessment/outputs/final_submission/`:
   - `adapter/` (`adapter_config.json`, `adapter_model.safetensors`)
   - `predictions_test.jsonl`
   - `cleaning_log.csv`
   - `results_bundle.json` and `outputs/error_analysis_failures.json` (used to populate `report.md`)
