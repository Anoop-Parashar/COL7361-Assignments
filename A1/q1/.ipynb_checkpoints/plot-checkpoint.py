import matplotlib.pyplot as plt 
import sys 
x_ap_fp = [90 ,50, 25, 10, 5] #support threshold 
y_ap = list(map(float, sys.argv[2].split())) #runtime 
y_fp = list(map(float, sys.argv[3].split())) 
outdir = sys.argv[1] 
plt.figure(figsize=(10, 7))
plt.plot(x_ap_fp, y_ap, marker='o', color = 'blue',label="Apriori",markersize=8) 
plt.plot(x_ap_fp, y_fp, marker='s', color = 'orange', label="FP-Growth", linestyle="--",markersize=8) 

plt.margins(y=0.3)

for xi, yi in zip(x_ap_fp, y_ap): 
    plt.annotate(f"{yi:.2f}", 
                 (xi, yi), 
                 textcoords="offset points", 
                 xytext=(0, 15), 
                 ha='center', 
                 va='bottom', 
                 color='blue', 
                 fontsize=9,
                 fontweight='bold') 


for xi, yi in zip(x_ap_fp, y_fp): 
    plt.annotate(f"{yi:.2f}", 
                 (xi, yi), 
                 textcoords="offset points", 
                 xytext=(0, -25), 
                 ha='center', 
                 va='top', 
                 color='orange', 
                 fontsize=9,
                 fontweight='bold')

plt.title('Support threshold(%) Vs Execution Time (seconds)', fontsize=14) 
plt.xlabel('Support threshold (%)', fontsize=12) 
plt.ylabel('Execution Time (seconds)', fontsize=12) 
plt.legend(loc='upper right') 
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(outdir+"/plot.png")