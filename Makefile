.PHONY: install dev test lint format run clean docker

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

run:
	streamlit run src/viz/app.py

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

docker:
	docker build -t signaling-game .
	docker run -p 8501:8501 signaling-game
