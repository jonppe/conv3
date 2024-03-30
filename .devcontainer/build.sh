#!/bin/sh
mkdir -p ./temp/root

podman build  .. -f .devcontainer/Dockerfile_training -t mambara
