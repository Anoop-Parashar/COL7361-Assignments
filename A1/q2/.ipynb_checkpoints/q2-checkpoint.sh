mkdir -p $5
rm gspan_time.txt 2> /dev/null
rm fsg_time.txt 2> /dev/null
rm gaston_time.txt 2> /dev/null

python q2.py $@

min_supports=(0.05, 0.10, 0.25, 0.50, 0.95)

# Executing gspan for various supports
for sup in ${min_supports[@]}; do
	SUP_INT=$(awk -v fl="$sup" 'BEGIN {printf "%d", fl * 100}')
	SUP_GASTON=$(awk -v fl="$sup" 'BEGIN {printf "%d", fl * 64110}')

	START=$(date +%s%N)
	$2 fsg_graphs.txt -s ${SUP_INT}.0 
	END=$(date +%s%N)
	DURATION=$(((END-START) / 1000000000))
	echo $DURATION >> fsg_time.txt
	mv fsg_graphs.fp ${5}/fsg"$SUP_INT"

	START=$(date +%s%N)
	$1 -f gspan_graphs.txt -s $sup -o -i 
	END=$(date +%s%N)
	DURATION=$(((END-START) / 1000000000))
	echo $DURATION >> gspan_time.txt
	mv gspan_graphs.txt.fp ${5}/gspan"$sup_int"

	START=$(date +%s%N)
	$3 $SUP_GASTON gaston_graphs.txt gaston_graphs.fp
	END=$(date +%s%N)
	DURATION=$(((END-START) / 1000000000))
	echo $DURATION >> gaston_time.txt
	mv gaston_graphs.fp ${5}/gaston"$SUP_INT"
	
done

# Plot TTEs of all methods
python plot.py $5 

# Clean up files created during program execution
rm gspan_graphs.txt 2> /dev/null
rm fsg_graphs.txt 2> /dev/null
rm gaston_graphs.txt 2> /dev/null

rm gspan_time.txt 2> /dev/null
rm fsg_time.txt 2> /dev/null
rm gaston_time.txt 2> /dev/null
