python fsm.py $1

./gSpan -f gspan_graphs.txt -s 0.2 -o -i 

python identify.py gspan_graphs.txt.fp $1 $2

#Cleanup

rm gspan_graphs.txt
rm gspan_graphs.txt.fp

