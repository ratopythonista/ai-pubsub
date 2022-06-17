.ONESHELL: # Applies to every targets in the file!

build:
	cd pubsub
	rm -rf dist/
	python -m build
	cd ..
	rm service_a/jai_pubsub-0.0.1-py3-none-any.whl
	mv pubsub/dist/jai_pubsub-0.0.1-py3-none-any.whl service_a
	docker-compose build 

up:
	docker-compose up