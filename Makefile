.PHONY : test
test:
	python -m pytest --disable-warnings .


.PHONY : lint
lint:
	ruff check --fix .
	blue -l 128 .
