import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import urllib.request
import json
import sys
from yellowbrick.cluster import KElbowVisualizer

arg1 = sys.argv[1]

def fetch_dataset(dataset_num):
    url = f"http://hulk.cse.iitd.ac.in:3000/dataset?student_id=siy257573&dataset_num={dataset_num}"
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode('utf-8')
        data = json.loads(raw_data)
        return np.array(data["X"])

def run_kmeans_and_get_optimal_k(data):
    inertia = [0] * 16
    for k in range(1, 16):
        km = KMeans(n_clusters=k, n_init="auto").fit(data)
        inertia[k] = km.inertia_
    model = KElbowVisualizer(KMeans(n_init="auto"), k=(1, 16))
    model.fit(data)
    optimal_k = model.elbow_value_
    plt.close('all')
    plt.clf()
    return inertia, optimal_k

def plot_elbow(ax, inertia, optimal_k, title):
    k_values = list(range(1, 16))
    ax.plot(k_values, inertia[1:], marker='o', label="Inertia vs K")
    ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=1, label=f'Optimal K={optimal_k}')
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (Sum of Squared Distances)")
    ax.set_title(title)
    ax.legend()

# Mode 1: dataset number given - run both datasets 
if arg1.isnumeric():
    data1 = fetch_dataset(1)
    data2 = fetch_dataset(2)

    inertia1, optimal_k1 = run_kmeans_and_get_optimal_k(data1)
    inertia2, optimal_k2 = run_kmeans_and_get_optimal_k(data2)

    plt.close('all')
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_elbow(ax1, inertia1, optimal_k1, title="Dataset 1 - Elbow Plot")
    plot_elbow(ax2, inertia2, optimal_k2, title="Dataset 2 - Elbow Plot")
    plt.tight_layout()
    plt.savefig('plot.png')
    plt.close()

    print(optimal_k1, optimal_k2)

# Mode 2: .npy file given  run single dataset 
else:
    data = np.load(arg1)

    inertia, optimal_k = run_kmeans_and_get_optimal_k(data)

    plt.close('all')
    plt.clf()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plot_elbow(ax, inertia, optimal_k, title="Input Dataset - Elbow Plot")
    plt.tight_layout()
    plt.savefig('plot.png')
    plt.close()

    print(optimal_k)