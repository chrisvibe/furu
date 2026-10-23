#!/bin/bash

image="boolean_reservoir"
remote="ANONYMIZED/boolean_reservoir"
version="4.1"
build_context="."

setup_directories() {
    mkdir -p ../data
    mkdir -p ../out
    echo "Ensured ../data and ../out directories exist."
}

pull_or_build_image() {
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed on this machine (this is normal for cluster login nodes!)."
        echo "Use './setup.sh conda' instead."
        return 1
    fi
    echo "Pulling $remote:$version from Docker Hub..."
    if docker pull $remote:$version; then
        docker tag $remote:$version $image:$version
        echo "Done. Image available as $image:$version"
    else
        echo "Pull failed — building locally (this will take a while)..."
        docker build -t $image:$version $build_context
    fi
}

setup_conda_env() {
    local lockfile="src/environment.lock.yaml"
    local envname="boolean_reservoir"
    echo "Setting up Conda environment from lock file..."
    if conda env list | grep -q "^$envname "; then
        echo "Environment '$envname' exists — updating..."
        conda env update -f $lockfile --prune
    else
        echo "Creating environment '$envname'..."
        conda env create -f $lockfile
    fi
    echo "Done. Activate with: conda activate $envname"
}

setup_directories

if [ "$1" == "conda" ]; then
    setup_conda_env
elif [ "$1" == "docker" ]; then
    pull_or_build_image
elif [ "$1" == "all" ]; then
    setup_conda_env
    pull_or_build_image
else
    echo "Usage: ./setup.sh [conda|docker|all]"
fi
