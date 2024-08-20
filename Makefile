
install:
	pip install --upgrade pip && pip install -r requirements.txt

test:
	python -m pytest -vv *.py

debug:
	python -m pytest -vv --pdb *.py

format:
	black --check . --exclude "/(label_processing|\.git|\.venv|docs)/"

lint:
	ruff check .

all: install test debug format lint