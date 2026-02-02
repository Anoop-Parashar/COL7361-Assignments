import matplotlib.pyplot as plt
import sys


def main():
    x = [5, 10, 25, 50, 95]
    t_gspan = []
    t_fsg = []
    t_gaston = []

    with open("gspan_time.txt", "r") as f:
        for line in f:
            t_gspan.append(float(line.strip()))

    with open("fsg_time.txt", "r") as f:
        for line in f:
            t_fsg.append(float(line.strip()))

    with open("gaston_time.txt", "r") as f:
        for line in f:
            t_gaston.append(float(line.strip()))

    print("Execution times for fsg:", t_fsg)
    print("Execution times for gspan:", t_gspan)
    print("Execution times for gaston:", t_gaston)

    plt.figure(figsize=(10, 10))
    plt.plot(x, t_gspan, color="b")
    plt.plot(x, t_fsg, color="g")
    plt.plot(x, t_gaston, color="r")

    plt.xlabel("Minimum Support (in %)")
    plt.ylabel("Time to Mine Frequent Subgraphs (in s)")
    plt.legend(["gSpan", "fsg", "gaston"])

    plt.savefig(f"{sys.argv[1]}/plot.png")


if __name__ == "__main__":
    main()
