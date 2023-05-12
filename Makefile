.PHONY : test
test:
	python -m pytest --disable-warnings .


.PHONY : lint
lint:
	blue -l 128 .
	ruff check --fix .
