#!/bin/bash

echo "Installing required packages..."

pip install numpy tqdm networkx python-igraph rustworkx

chmod +x ./gSpan
