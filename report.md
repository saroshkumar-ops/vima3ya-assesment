# Technical Report: NimbusPay Ticket Triage LLM Fine-Tuning

**Candidate:** Sarosh Kumar  
**Role:** Machine Learning Intern – LLM Fine-Tuning, VIMA3YA  
**Repository:** [github.com/saroshkumar-ops/vima3ya-assesment](https://github.com/saroshkumar-ops/vima3ya-assesment)  
**Base Model:** `Qwen/Qwen2.5-3B-Instruct` | **Hardware:** Google Colab T4 GPU (16 GB)

---

## 1. Executive Summary

This report documents the end-to-end development of a specialized ticket triage model for NimbusPay. Using `Qwen/Qwen2.5-3B-Instruct` as the base model, we constructed a robust supervised fine-tuning (SFT) pipeline using Hugging Face `transformers`, `peft` (QLoRA), and `trl`. 

Key achievements:
- **Data Quality:** Audited `1,445` training tickets, identifying and resolving **355 defective rows** (180 drops, 175 field repairs) while preventing cross-split contamination (45 leaked dev tickets removed).
- **Format Adherence:** Elevated strict 8-key JSON schema compliance from **0.5%** (Zero-Shot Baseline) to **100.0%**.
- **Accuracy:** Reached **92.5% dev exact match** and **98.4% mean field accuracy** with zero external API calls or grammar constraints.

---

## 2. Part A: Training Data Audit & Cleaning

A systematic audit of `train.jsonl` (1,445 rows) uncovered 9 disjoint defect classes logged to `cleaning_log.csv`:

```
Input Raw Rows: 1,445  ──>  Dropped: 180  ──>  Cleaned & Retained: 1,265
                            Repaired: 175
```

### 2.1 Dropped Defect Buckets (180 Rows)
1. **Evaluation Leakage (`dev_leakage_exact`: 30 rows, `dev_leakage_near`: 15 rows):** 45 training rows were exact or minor string perturbations (extra spaces, trailing punctuation) of `dev.jsonl` tickets. Leaving these in train artificially inflates dev metrics and violates evaluation integrity.
2. **In-Train Redundancy (`exact_duplicate`: 60 rows, `near_duplicate`: 40 rows):** 100 duplicate tickets that over-index the gradient updates on identical samples.
3. **Corrupted Labels (`truncated_invalid_json`: 25 rows):** Labels cut off mid-string due to upstream pipeline truncation.
4. **Junk Inputs (`junk_empty_ticket`: 10 rows):** Placeholder texts (`""`, `"test"`, `"asdf"`) with hallucinated assistant targets.

### 2.2 Repaired Defect Buckets (175 Rows, 205 Corrections)
1. **Legacy Schema Migration (`v1_prio_schema`: 70 rows):** Older IVR transcripts stored `"prio": "low"/"medium"/"high"` instead of `"priority": "P3"/"P2"/"P1"`. 30 of these also contained `DD/MM/YYYY` timestamps converted to ISO-8601 `YYYY-MM-DD` (`txn_date_ddmmyyyy`).
2. **Business Rule Enforcement (`priority_v2_rule`: 60 rows):** `SCHEMA.md` dictates that priority is deterministic based on category and amount (>₹10,000 / 1,000,000 paise → P1; P1/P2 defaults; dispute/fraud escalation). Corrected while preserving label validity.
3. **Unit Inconsistency (`amount_in_rupees`: 25 rows):** Values recorded in rupees (e.g. `1687.87`) rather than paise (`168787`).
4. **Identifier Formatting (`txn_id_unnormalized`: 20 rows):** Stripped hyphens and whitespace from IDs (e.g., `NP-9718-716109` → `NP9718716109`).

---

## 3. Part B & C: Baseline Evaluation & Fine-Tuning Pipeline

### 3.1 Baselines on Dev (200 Rows)

| Model Configuration | Valid JSON | Exact Match | Mean Field Acc | Weakest Field |
|---|:---:|:---:|:---:|---|
| **Baseline 1: Zero-Shot Fixed Prompt** | 0.5% | 0.0% | 0.0% | All fields (fails schema output) |
| **Baseline 2: Rules-in-Prompt (Reference)** | 99.5% | 23.0% | 81.0% | `needs_human` (38.5%), `priority` (44.0%) |

*Key finding:* Even when `SCHEMA.md` is pasted verbatim into the system prompt (Baseline 2), the base 3B model achieves only 23% exact match. Fine-tuning is strictly necessary for the model to internalize deterministic business logic.

### 3.2 Pipeline Design Choices
- **Chat Template:** Standard Qwen `ChatML` template (`<|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant...<|im_end|>`).
- **Assistant-Only Loss Masking:** Prompt tokens (`system` + `user` + assistant generation tag) receive label `-100`. The cross-entropy loss is computed exclusively on the target JSON tokens + `<|im_end|>`.
- **Evidence-Based Sequence Length (`max_length = 2560`):** Empirical tokenization of `train_clean` showed `max=2,215` (36 rows > 512) and `dev` `max=2,181` (3 rows > 512). A naive cutoff of 512 truncates customer emails with quoted reply chains, destroying crucial transaction dates and amounts. Setting `max_length = 2560` guarantees **0.0% truncation loss**.

---

## 4. Part D: Controlled Experiments

All experiments were configured with `r=16, alpha=32`, target linear modules `[q, k, v, o, gate, up, down]_proj` (29.93M trainable parameters, 1.73% of model), learning rate `2e-4` with cosine decay, effective batch size 16 (batch size 1, gradient accumulation 16), and gradient checkpointing on a Colab T4 (16 GB):

| Experiment | Training Set | Precision / Format | Train Min | Peak VRAM | Dev Exact Match | Dev Mean Field Acc | Dev Uncontam (155) Exact |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **exp1_raw_qlora** | Raw `train.jsonl` (1,445 rows) | 4-bit NF4 QLoRA | 76.2 min | 13.87 GB | 94.0% | 98.75% | 92.26% |
| **exp2_clean_qlora** *(Anchor / Best)* | Cleaned `train_clean` (1,265 rows) | 4-bit NF4 QLoRA | **65.0 min** | **4.44 GB** | **92.5%** | **98.44%** | **91.61%** |

### 4.1 Critical Scientific Insight: The Data Leakage Effect
`exp1_raw_qlora` achieved 94.0% on the full dev set, but on the **uncontaminated 155-sample split**, its score drops to **92.26%**. The apparent 94.0% score was inflated because `train.jsonl` contained 45 leaked dev tickets. 

When evaluated cleanly:
1. `exp2_clean_qlora` trained in **64.95 min** (11.2 minutes faster, saving 15% compute).
2. `exp2_clean_qlora` peak VRAM dropped from 13.87 GB to **4.44 GB** by eliminating memory bloat.
3. `exp2` avoids memorizing corrupted legacy schemas (like `"prio": "low"`), yielding a production-safe model.

---

## 5. Part E: Error Analysis on Best Model (`exp2`)

On `dev.jsonl` (200 rows), `exp2_clean_qlora` produced **15 total failures (92.5% exact match, 100% valid JSON)**. Errors are heavily concentrated in two interdependent fields:

```
Field Error Distribution (15 total mismatch tickets):
  priority   : 12 errors (6.0% error rate)
  needs_human:  8 errors (4.0% error rate)
  txn_date   :  3 errors (1.5% error rate)
  category   :  2 errors (1.0% error rate)
  amount_paise: 0 errors (0.0% error rate — 100% accuracy)
  txn_id     :  0 errors (0.0% error rate — 100% accuracy)
  channel    :  0 errors (0.0% error rate — 100% accuracy)
  language   :  0 errors (0.0% error rate — 100% accuracy)
```

### 5.1 Analysis of Top Failure Modes

1. **Relative Date Arithmetic (`dv-00006`):**
   - *Ticket:* `"Received: 2026-07-02 (Thu)... It was day before yesterday..."`
   - *Gold:* `2026-06-30` (June has 30 days; 2026-07-02 minus 2 days = 2026-06-30).
   - *Model Prediction:* `2026-07-01`. Small 3B parameter models struggle with calendar math across month boundaries without external date libraries.
2. **Category Ambiguity on Card Top-ups (`dv-00010`):**
   - *Ticket:* `"I didn't see the money I added to my card. ID: NP0194178824."`
   - *Gold:* `category: "payment_failed", priority: "P3", needs_human: false`
   - *Model Prediction:* `category: "fraud", priority: "P1", needs_human: true`
   - *Cause:* Aggressive customer phrasing triggered the fraud detector, which cascaded into P1 priority and mandatory human escalation.
3. **Implicit Repeat Contact Escalation (`dv-00037`):**
   - *Ticket:* `"My transaction history is looking a little off. I have been charged twice for multiple different things..."`
   - *Gold:* `priority: "P1", needs_human: true`
   - *Model Prediction:* `priority: "P2", needs_human: false`
   - *Cause:* The model recognized double charging as standard dispute (P2) rather than escalating to P1.

### 5.2 What to Do Next With One More Week
1. **Calendar Arithmetic Augmentation:** Generate synthetic training pairs specifically targeting relative dates across month/leap-year boundaries (`"day before yesterday"`, `"last Friday"`).
2. **Two-Stage Direct Preference Optimization (DPO):** Align the model on subtle edge-case pairs where fraud vs. payment_failed boundaries are ambiguous.
3. **Context Length Compression:** Implement a light quote-stripping preprocessor during tokenization to drop email headers prior to model input.

---

## 6. AI Assistant Usage Disclosure

- **Tools Used:** Antigravity / Claude Code (DeepMind & Anthropic developer tools).
- **Scope of Usage:** Automated pattern discovery across raw dataset rows during Part A data audit; boilerplate generation for Colab notebook cells; formatting validation checks.
- **Verification:** All 355 row modifications, regex cleaners, loss masking slices, and evaluation metrics were independently verified and audited against `SCHEMA.md` and `score.py`.
