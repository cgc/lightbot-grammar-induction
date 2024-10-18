test:
	python -m pytest ${PYTEST_ARGS}

watch-test:
	find . -path ./env -prune -o -name '*.py' | entr sh -c "make test"
