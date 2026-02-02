#!/usr/bin/env python3
import sys
import numpy as np


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 generate_candidates.py <db.npy> <query.npy> <out.dat>")
        sys.exit(1)

    db_path = sys.argv[1]
    query_path = sys.argv[2]
    out_path = sys.argv[3]

    DB = np.load(db_path)
    Q = np.load(query_path)

    num_queries = Q.shape[0]

    with open(out_path, "w") as f:
        for qi in range(num_queries):
            qvec = Q[qi]

            
            mask = np.all(DB >= qvec, axis=1)

            candidates = np.where(mask)[0]

            f.write(f"q # {qi+1}\n")
            f.write("c # " + " ".join(map(str, candidates)) + "\n")

    print("generate_candidates.py completed")
    print("Output saved to:", out_path)


if __name__ == "__main__":
    main()
