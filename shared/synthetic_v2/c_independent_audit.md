# C — Independent Selection and Statistics Audit

## Environment

- Date: 2026-11-15
- OS: Windows 10
- Python: 3.12.15
- Execution: Local CPU, Python standard library

## 1. Data Integrity

- Input: `training_bundle.zip`
- SHA-256: `4e47d820939622039a769cb97211ae6aac5c3f0885797e281e59b1d52256121f`
- Result: **PASS**

## 2. CLIP Selection Audit

Script: `code/audits/02_clip_audit.py`

Verified:
- 2,000 candidate records and matching CLIP score records.
- 200 candidates per class.
- Class labels, image hashes and stored confidence values.
- CLIP ranking consistency.
- Random and CLIP selection counts (50/150 per class).
- Selection nesting and duplicate ID checks.

Result: **ALL CHECKS PASS**

Scope: Validation of stored records only. CLIP inference and random permutation generation were not independently rerun.

## 3. Statistics Protocol Audit

Script: `code/audits/03_stats_audit.py`

Verified using simulated test data:
- Top-1 accuracy calculation.
- Complete 5-condition × 3-seed experiment matrix.
- Mean and sample standard deviation (`ddof=1`).
- Same-seed paired improvements in percentage points.

Result: **ALL TOY-DATA PROTOCOL CHECKS PASS**

This test validates calculation rules, not actual classifier performance.

## 4. Evidence and Next Steps

Both scripts were executed locally with exit code 0.

Original PowerShell transcripts are retained locally and are not uploaded because they contain personal filesystem paths.

Pending:
- G4 official test-set evaluation on A's GPU.
- CLIP zero-shot evaluation.
- Independent recalculation using official predictions.
- Final figures and cost analysis.
