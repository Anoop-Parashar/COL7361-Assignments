#!/bin/bash
# forest_fire.sh
# Usage:
#   bash forest_fire.sh <graph> <seeds> <output> <k> <n_random_instances> <hops>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/forest_fire.py" "$1" "$2" "$3" "$4" "$5" "$6"