#!/usr/bin/env bash

./build.sh

echo "Installing..."

python3 setup.py install --record installed_files.txt

echo "----- Installation finished -----"
