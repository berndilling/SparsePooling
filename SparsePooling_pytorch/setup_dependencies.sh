#!/usr/bin/env bash
echo "Make sure conda is installed."
echo "Installing environment:"

if [ "$(uname)" == "Darwin" ]; then
    conda env create -f env_mac.yml || conda env update -f env_mac.yml || exit
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    conda env create -f env.yml || conda env update -f env.yml || exit
fi

conda activate sparsepooling_pytorch

