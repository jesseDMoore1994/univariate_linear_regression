.PHONY: all
all: test

.PHONY: test
test: .test_docker
	@docker run -it --rm -v $(shell pwd):/app -w /app --user $(shell id -u):$(shell id -g) univariate_linear_regression python setup.py test 

.PHONY: dist
dist: test
	@docker run -it --rm -v $(shell pwd):/app -w /app --user $(shell id -u):$(shell id -g) univariate_linear_regression python setup.py sdist 

.PHONY: version
version: 
	@docker run -it --rm -v $(shell pwd):/app -w /app --user $(shell id -u):$(shell id -g) univariate_linear_regression python setup.py --version

.test_docker:
	@docker build -t univariate_linear_regression -f Dockerfile.test .
	@touch $@

.PHONY: clean
clean:
	@docker rmi -f univariate_linear_regression
	@rm -rf .test_docker dist
