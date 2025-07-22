# ======================
# Development & Quality
# ======================
install-dev:
	uv sync --dev

lint:
	uv run ruff check src/ data/

format:
	uv run black --check src/ data/

type-check:
	uv run mypy src/ data

test:
	uv run pytest -s -m integration

validate: lint type-check

run: 
	uv run python -m streamlit run ./src/chatbot/app.py