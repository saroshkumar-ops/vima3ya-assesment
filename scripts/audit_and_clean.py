"""Part A: Audit and clean NimbusPay training data (train.jsonl) strictly against SCHEMA.md.

Usage:
    python scripts/audit_and_clean.py \
        --input ../training_assessment/candidate_pack/data/train.jsonl \
        --dev ../training_assessment/candidate_pack/data/dev.jsonl \
        --output-clean work_data/train_clean.jsonl \
        --output-log cleaning_log.csv
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

KEYS = [
    "category",
    "priority",
    "amount_paise",
    "txn_date",
    "txn_id",
    "channel",
    "language",
    "needs_human",
]
ALLOWED_CATEGORIES = {
    "payment_failed",
    "refund",
    "fraud",
    "account_access",
    "kyc",
    "offers",
    "other",
}
ALLOWED_PRIORITIES = {"P1", "P2", "P3"}
ALLOWED_CHANNELS = {"upi", "card", "netbanking", "wallet", None}
ALLOWED_LANGUAGES = {"en", "hi-en"}
V1_PRIO_MAP = {"high": "P1", "medium": "P2", "low": "P3"}

LEGAL_PAT = re.compile(
    r"\b(rbi|ombudsman|consumer\s*court|lawyer|legal\s*notice|legal\s*action)\b",
    re.I,
)
AMT_PAT = re.compile(
    r"(?:(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)(?:\s*(k|lakh|lakhs)\b)?|"
    r"\b([\d,]+(?:\.\d+)?)\s*(rupees|rupaye|k|lakh|lakhs)\b)",
    re.I,
)


def normalize_for_dedup(user_text: str) -> str:
    """Normalize trivial trailing-token and whitespace perturbations for near-duplicate detection."""
    s = re.sub(r"\s+", " ", user_text).strip()
    s = re.sub(r"\s+Thx$", "", s)
    s = re.sub(r"\s+\.$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def non_quoted_body(user_text: str) -> str:
    """Extract ticket body excluding quoted earlier email lines starting with '>'."""
    body = user_text.split("\n\n", 1)[1] if "\n\n" in user_text else user_text
    return "\n".join(line for line in body.splitlines() if not line.lstrip().startswith(">"))


def extract_money_candidates(text: str) -> list[tuple[float, int, int]]:
    """Return list of (rupee_float, expected_paise_int, truncated_rupee_int) from text."""
    cands = []
    for m in AMT_PAT.finditer(text):
        g1, u1, g2, u2 = m.groups()
        num_str = (g1 or g2).replace(",", "")
        unit = (u1 or u2 or "").lower()
        try:
            val = float(num_str)
            if unit == "k":
                val *= 1000
            elif unit.startswith("lakh"):
                val *= 100000
            cands.append((val, int(round(val * 100)), int(val)))
        except ValueError:
            continue
    return cands


def compute_expected_priority(category: str, amount_paise: int | None, non_quoted: str) -> str:
    """Apply SCHEMA.md v2 priority rules strictly in order."""
    has_legal = bool(LEGAL_PAT.search(non_quoted))
    if category == "fraud" or (amount_paise is not None and amount_paise >= 5_000_000) or has_legal:
        return "P1"
    if (
        category in {"payment_failed", "refund"} and amount_paise is not None and amount_paise >= 500_000
    ) or category == "account_access":
        return "P2"
    return "P3"


def validate_clean_record(obj: dict) -> list[str]:
    """Verify a record conforms 100% to SCHEMA.md types and allowed values."""
    errs = []
    if list(obj.keys()) != KEYS:
        errs.append(f"unexpected_keys:{list(obj.keys())}")
    if obj.get("category") not in ALLOWED_CATEGORIES:
        errs.append(f"bad_category:{obj.get('category')}")
    if obj.get("priority") not in ALLOWED_PRIORITIES:
        errs.append(f"bad_priority:{obj.get('priority')}")
    amt = obj.get("amount_paise")
    if amt is not None and type(amt) is not int:
        errs.append(f"bad_amount_type:{type(amt).__name__}")
    tdate = obj.get("txn_date")
    if tdate is not None and (type(tdate) is not str or not re.match(r"^\d{4}-\d{2}-\d{2}$", tdate)):
        errs.append(f"bad_txn_date:{tdate}")
    tid = obj.get("txn_id")
    if tid is not None and (type(tid) is not str or not re.match(r"^NP\d{10}$", tid)):
        errs.append(f"bad_txn_id:{tid}")
    if obj.get("channel") not in ALLOWED_CHANNELS:
        errs.append(f"bad_channel:{obj.get('channel')}")
    if obj.get("language") not in ALLOWED_LANGUAGES:
        errs.append(f"bad_language:{obj.get('language')}")
    if type(obj.get("needs_human")) is not bool:
        errs.append(f"bad_needs_human:{obj.get('needs_human')}")
    if obj.get("category") in {"account_access", "kyc", "other"}:
        if any(obj.get(k) is not None for k in ("amount_paise", "txn_date", "txn_id", "channel")):
            errs.append("non_null_txn_field_for_non_txn_category")
    return errs


def audit_and_clean(input_path: Path, dev_path: Path | None, output_clean: Path, output_log: Path) -> None:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_exact: dict[str, str] = {}
    dev_near: dict[str, str] = {}
    if dev_path is not None and dev_path.exists():
        for drow in (json.loads(l) for l in dev_path.read_text(encoding="utf-8").splitlines() if l.strip()):
            du = drow["messages"][1]["content"]
            dev_exact[du] = drow["id"]
            dev_near[normalize_for_dedup(du)] = drow["id"]

    seen_exact: dict[str, str] = {}
    seen_near: dict[str, str] = {}

    log_entries: list[tuple[str, str, str]] = []
    cleaned_rows: list[dict] = []
    reason_counts: Counter = Counter()

    for row in rows:
        rid = row["id"]
        msgs = row["messages"]
        user_text = msgs[1]["content"]
        raw_label = msgs[-1]["content"]
        body = (user_text.split("\n\n", 1)[1] if "\n\n" in user_text else user_text).strip()

        # 1. Check truncated / invalid JSON
        try:
            obj = json.loads(raw_label.strip())
            if not isinstance(obj, dict):
                raise ValueError("Assistant output is not a JSON object")
        except Exception:
            log_entries.append((rid, "drop", "truncated_invalid_json"))
            reason_counts["drop:truncated_invalid_json"] += 1
            continue

        # 2. Check junk / empty / placeholder tickets
        if len(body) < 20 or re.fullmatch(r"(test(\s+test)*(\s+\d+)?|asdf(\s+asdf)*)", body, re.I):
            log_entries.append((rid, "drop", f"junk_empty_ticket (body={body!r})"))
            reason_counts["drop:junk_empty_ticket"] += 1
            continue

        # 3. Check dev leakage (never train on dev rows)
        norm_u = normalize_for_dedup(user_text)
        if user_text in dev_exact:
            log_entries.append((rid, "drop", f"dev_leakage_exact:{dev_exact[user_text]}"))
            reason_counts["drop:dev_leakage_exact"] += 1
            continue
        if norm_u in dev_near:
            log_entries.append((rid, "drop", f"dev_leakage_near:{dev_near[norm_u]}"))
            reason_counts["drop:dev_leakage_near"] += 1
            continue

        # 4. Check exact & near-duplicates within train
        if user_text in seen_exact:
            log_entries.append((rid, "drop", f"exact_duplicate_of:{seen_exact[user_text]}"))
            reason_counts["drop:exact_duplicate"] += 1
            continue
        if norm_u in seen_near:
            log_entries.append((rid, "drop", f"near_duplicate_of:{seen_near[norm_u]}"))
            reason_counts["drop:near_duplicate"] += 1
            continue
        seen_exact[user_text] = rid
        seen_near[norm_u] = rid

        # 4. Deterministic fixes against SCHEMA.md
        fix_reasons: list[str] = []

        # 4a. Legacy v1 schema ("prio" -> "priority")
        if "prio" in obj:
            old_prio = obj.pop("prio")
            obj["priority"] = V1_PRIO_MAP[old_prio]
            fix_reasons.append(f"v1_prio_schema:{old_prio}->{obj['priority']}")
            reason_counts["fix:v1_prio_schema"] += 1

        # 4b. Legacy DD/MM/YYYY date format -> YYYY-MM-DD
        tdate = obj.get("txn_date")
        if tdate is not None and re.match(r"^\d{2}/\d{2}/\d{4}$", str(tdate)):
            d, m, y = tdate.split("/")
            obj["txn_date"] = f"{y}-{m}-{d}"
            fix_reasons.append(f"txn_date_ddmmyyyy:{tdate}->{obj['txn_date']}")
            reason_counts["fix:txn_date_ddmmyyyy"] += 1

        # 4c. Unnormalized txn_id -> uppercase NP + 10 digits
        tid = obj.get("txn_id")
        if tid is not None and not re.match(r"^NP\d{10}$", str(tid)):
            norm_tid = re.sub(r"[^A-Za-z0-9]", "", str(tid)).upper()
            assert re.match(r"^NP\d{10}$", norm_tid), f"{rid}: failed txn_id fix {tid!r} -> {norm_tid!r}"
            fix_reasons.append(f"txn_id_unnormalized:{tid}->{norm_tid}")
            obj["txn_id"] = norm_tid
            reason_counts["fix:txn_id_unnormalized"] += 1

        # 4d. amount_paise stored in rupees instead of paise
        nq = non_quoted_body(user_text)
        cat = obj["category"]
        amt = obj.get("amount_paise")
        if cat not in {"account_access", "kyc", "other"} and amt is not None:
            cands = extract_money_candidates(nq)
            if all(paise != amt for _, paise, _ in cands):
                matched = [paise for _, paise, trunc_r in cands if trunc_r == amt]
                assert matched, f"{rid}: unexplained amount mismatch amt={amt}, cands={cands}"
                fix_reasons.append(f"amount_in_rupees:{amt}->{matched[0]}")
                obj["amount_paise"] = matched[0]
                reason_counts["fix:amount_in_rupees"] += 1

        # 4e. Corrupted priority under SCHEMA.md v2 rules
        exp_prio = compute_expected_priority(cat, obj.get("amount_paise"), nq)
        if obj["priority"] != exp_prio:
            fix_reasons.append(f"priority_v2_rule:{obj['priority']}->{exp_prio}")
            obj["priority"] = exp_prio
            reason_counts["fix:priority_v2_rule"] += 1

        # Canonicalize key order
        clean_obj = {k: obj[k] for k in KEYS}
        errs = validate_clean_record(clean_obj)
        assert not errs, f"{rid}: validation failed after cleaning: {errs}"

        if fix_reasons:
            log_entries.append((rid, "fix", "; ".join(fix_reasons)))

        cleaned_rows.append(
            {
                "id": rid,
                "messages": [
                    msgs[0],
                    msgs[1],
                    {"role": "assistant", "content": json.dumps(clean_obj, ensure_ascii=False)},
                ],
            }
        )

    output_clean.parent.mkdir(parents=True, exist_ok=True)
    with output_clean.open("w", encoding="utf-8") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_log.parent.mkdir(parents=True, exist_ok=True)
    with output_log.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "action", "reason"])
        writer.writerows(log_entries)

    drops = sum(1 for _, act, _ in log_entries if act == "drop")
    fixes = sum(1 for _, act, _ in log_entries if act == "fix")
    print(f"Input rows:   {len(rows)}")
    print(f"Dropped rows: {drops}")
    print(f"Fixed rows:   {fixes}")
    print(f"Clean rows:   {len(cleaned_rows)}")
    print("\nBreakdown by issue:")
    for k, v in sorted(reason_counts.items()):
        print(f"  {k:32s} {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--output-clean", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    args = parser.parse_args()
    audit_and_clean(args.input, args.dev, args.output_clean, args.output_log)
