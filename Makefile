check:
	pytest

fact-baseline:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/eval_clean_detection.py input/detext_examples --manifest input/detext_examples/fact_baseline.json --mask-dir input/detext_examples/mask --output result/fact-baseline/report.json --artifact-dir result/fact-baseline/previews

fact-baseline-formal:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/eval_clean_detection.py input/detext_examples --manifest input/detext_examples/fact_baseline.json --mask-dir input/detext_examples/mask --output result/fact-baseline/report.json --artifact-dir result/fact-baseline/previews --require-clean-git
