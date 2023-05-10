.PHONY : test
test:
	python -m pytest --disable-warnings .


.PHONY : lint
lint:
	isort .
	blue -l 128 .
	ruff check --fix .
