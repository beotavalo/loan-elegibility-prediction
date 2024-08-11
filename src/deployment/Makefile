# Makefile
.PHONY: build run push

build:
	docker build -t flask-app .

run:
	docker run -p 5000:5000 flask-app

push:
	docker tag flask-app botavalo/flask-app:latest
	docker push botavalo/flask-app:latest

pull:
	docker pull botavalo/flask-app:latest
