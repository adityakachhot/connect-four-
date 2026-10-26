.PHONY: setup test train evaluate ui report help

help:
	@echo "Available make targets:"
	@echo "  setup     - Install dependencies from pinned requirements.txt"
	@echo "  test      - Run full pytest test suite"
	@echo "  train     - Run Connect Four AlphaZero training pipeline"
	@echo "  evaluate  - Evaluate best checkpoints against minimax baseline"
	@echo "  ui        - Launch Streamlit interactive game UI"
	@echo "  report    - Generate report figures and match transcripts"

setup:
	pip install -r requirements.txt

test:
	pytest tests/

train:
	python src/train_connect_four.py

evaluate:
	python src/eval/eval_vs_minimax.py

ui:
	streamlit run src/ui/app.py

report:
	python src/scripts/generate_report_assets.py
