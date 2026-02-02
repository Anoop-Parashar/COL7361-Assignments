if [ "$#" -ne 2 ]; then
    echo "Usage: bash q1_2.sh <universal_itemset> <num_transactions>"
    exit 1
fi

UNIVERSAL_ITEMSET=$1
NUM_TRANSACTIONS=$2
OUTFILE="generated_transactions.dat"

python3 generate_dataset.py "$UNIVERSAL_ITEMSET" "$NUM_TRANSACTIONS" > "$OUTFILE"
