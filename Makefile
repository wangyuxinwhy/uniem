.PHONY : test
test:
	python -m pytest --disable-warnings .


.PHONY : lint
lint:
	blue -l 128 .
	ruff check --fix .
	pyright .

.PHONY : check-lint
check-lint:
	blue -l 128 --check .
	ruff check --exit-non-zero-on-fix --fix .
	pyright .

.PHONY : publish
publish:
	poetry publish --build
