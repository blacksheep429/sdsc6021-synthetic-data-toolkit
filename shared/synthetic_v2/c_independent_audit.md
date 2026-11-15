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

## Integration and current handoff

PR #2 is integrated into the main branch. A reran both scripts successfully; the shared [integration check](audits/team_merge_checks.json) records the verification. The statistics script was normalized to repository LF line endings without changing its logic. The original C execution environment and limits above remain as reported.

G4 has since completed on A's GPU. C's next step is independent analysis of the [official predictions](g4/README.md), followed by figures and interpretation. The pending G4 items above describe the original submission time.

The selection audit expects training_bundle.zip in its working directory. From the repository root, with the G3 asset downloaded:

```powershell
$projectRoot = (Get-Location).Path
Push-Location (Join-Path $projectRoot 'deliverables/g3_synthetic_v2')
python (Join-Path $projectRoot 'code/audits/02_clip_audit.py')
Pop-Location
python code/audits/03_stats_audit.py
```
