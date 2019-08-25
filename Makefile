

.PHONY: all
all: test build

.PHONY: test
test: .test_docker
	@docker run -it --rm -v $(shell pwd):/app -w /app --user $(shell id -u):$(shell id -g) univariate_linear_regression_test python setup.py test 

.PHONY: build
build: test .prod_docker

.PHONY: deploy
deploy: 
	@docker run -it --rm -v $(shell pwd):/app univariate_linear_regression

.PHONY: dist
dist: 
	@docker run -it --rm -v $(shell pwd):/app -w /app --user $(shell id -u):$(shell id -g) univariate_linear_regression_test python setup.py sdist 

.PHONY: shell
shell: 
	@docker run -it --rm -v $(shell pwd):/app -w /app --user $(shell id -u):$(shell id -g) univariate_linear_regression_test bash

.PHONY: version
version: 
	@docker run -it --rm -v $(shell pwd):/app -w /app --user $(shell id -u):$(shell id -g) univariate_linear_regression_test python setup.py --version

.test_docker:
	@docker build -t univariate_linear_regression_test -f Dockerfile.test .
	@touch $@

.prod_docker: dist
	@docker build -t univariate_linear_regression --build-arg VERSION=$(shell make version) -f Dockerfile.prod .
	@touch $@

.PHONY: clean
clean:
	-docker rmi -f univariate_linear_regression_test
	-docker rmi -f univariate_linear_regression
	-rm -rf .prod_docker .test_docker dist plot.png
