#!/bin/bash

APRIORI_PATH=$1
FPTREE_PATH=$2
DATASET_PATH=$3
OutputDir=$4
mkdir -p "$OutputDir"
TMP_TIME_FILE="$OutputDir/time.tmp"

thresholds=(90 50 25 10 5)

ap_run=()
fp_run=()
for i in "${thresholds[@]}"
do
    echo "Running Apriori at minimum support threshold $i%"
	mkdir -p "$OutputDir/ap$i"
	output_file="$OutputDir/ap${i}/output.txt"
	touch "$output_file"
    timeout 3600s /usr/bin/time -f "%e" --quiet -o "$TMP_TIME_FILE" "$APRIORI_PATH" -s"$i" "$DATASET_PATH" "$output_file"
    
    if [ ! -s "$TMP_TIME_FILE" ]; then
        duration=3600
    else
        duration=$(cat "$TMP_TIME_FILE")
    fi
	
    ap_run+=("$duration")
done
rm -f "$TMP_TIME_FILE"

TMP_TIME_FILE="$OutputDir/time.tmp"
touch "$TMP_TIME_FILE"

for i in "${thresholds[@]}"
do
    echo "Running FPTree at minimum support threshold $i%"
    mkdir -p "$OutputDir/fp$i"
	output_file="$OutputDir/fp${i}/output.txt"
	touch "$output_file"
    timeout 3600s /usr/bin/time -f "%e" --quiet -o "$TMP_TIME_FILE" "$FPTREE_PATH" -s"$i" "$DATASET_PATH" "$output_file"
    
    if [ ! -s "$TMP_TIME_FILE" ]; then
        duration=3600
    else
        duration=$(cat "$TMP_TIME_FILE")
    fi

    fp_run+=("$duration")
	
done
rm -f "$TMP_TIME_FILE"

python3 plot.py "$OutputDir" "${ap_run[*]}" "${fp_run[*]}"





