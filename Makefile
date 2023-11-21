test:
	python -m pytest ${PYTEST_ARGS}

watch-test:
	find . -name '*.py' | entr sh -c "make test"
