#!/bin/bash
DB=$1
QUERY=$2
OUT=$3

python3 generate_candidates.py "$DB" "$QUERY" "$OUT"
