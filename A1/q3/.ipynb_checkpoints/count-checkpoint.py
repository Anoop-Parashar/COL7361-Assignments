counts = []

with open("candidates.dat") as f:
    for line in f:
        if line.startswith("c #"):
            nums = line.strip().split()[2:]
            counts.append(len(nums))

print("Total Queries:", len(counts))
print("Min Candidates:", min(counts))
print("Max Candidates:", max(counts))
print("Avg Candidates:", sum(counts) / len(counts))

print("\nFirst 10 Queries:")
for i, c in enumerate(counts, 1):
    print(f"Query {i}: {c} candidates")
