import random
import sys

random.seed(42)

def generate_dataset(num_transactions, num_items):
    items = [str(i) for i in range(1, num_items + 1)]

   
    freq_items1 = items[:num_items//4]
    freq_items2 = items[num_items//4:2*num_items//4]
    freq_items3 = items[2*num_items//4:3*num_items//4]

    for _ in range(num_transactions):
        trans = set()

        
        for _ in range(5):  
            if random.random() < 0.9:
                trans.update(random.sample(freq_items1, random.randint(1, len(freq_items1))))
            if random.random() < 0.8:
                trans.update(random.sample(freq_items2, random.randint(1, len(freq_items2))))
            if random.random() < 0.6:
                trans.update(random.sample(freq_items3, random.randint(1, len(freq_items3))))

       
        if random.random() < 0.2:
            noise = items[3*num_items//4:]
            trans.update(random.sample(noise, random.randint(1, 5)))

        if not trans:
            trans.add(random.choice(items))

        print(" ".join(sorted(trans, key=int)))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_dataset.py <num_items> <num_transactions>")
    else:
        num_items = int(sys.argv[1])
        num_transactions = int(sys.argv[2])
        generate_dataset(num_transactions, num_items)