import pandas as pd

file_path = '../data/musiccaps-train-data.csv'

df = pd.read_csv(file_path)

aspects = set()
for aspect_list in df['aspect_list']:
    for aspect in aspect_list:
        aspects.add(aspect)

print(f"Number of aspects: {len(aspects)}")