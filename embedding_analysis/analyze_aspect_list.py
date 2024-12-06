import pandas as pd
import ast

def get_aspects():
    file_path = '../data/musiccaps-train-data.csv'

    df = pd.read_csv(file_path)

    aspects = set()
    for aspect_list in df['aspect_list'].apply(ast.literal_eval):
        for aspect in aspect_list:
            aspects.add(aspect)

    return aspects

def get_avg_num_aspects():
    file_path = '../data/musiccaps-train-data.csv'

    df = pd.read_csv(file_path)

    total = 0
    for aspect_list in df['aspect_list'].apply(ast.literal_eval):
        total += len(aspect_list)

    return total / len(df['aspect_list'])

def get_min_aspects():
    file_path = '../data/musiccaps-train-data.csv'

    df = pd.read_csv(file_path)

    min = 11
    for aspect_list in df['aspect_list'].apply(ast.literal_eval):
        if len(aspect_list) < min:
            min = len(aspect_list)

    return min

def get_max_aspects():
    file_path = '../data/musiccaps-train-data.csv'

    df = pd.read_csv(file_path)

    max = 0
    for aspect_list in df['aspect_list'].apply(ast.literal_eval):
        if len(aspect_list) > max:
            max = len(aspect_list)

    return max

def main():
    print(f"Number of aspects: {len(get_aspects())}")
    print(f"Average number of aspects: {get_avg_num_aspects()}")
    print(f"Min number of aspects: {get_min_aspects()}")
    print(f"Max number of aspects: {get_max_aspects()}")

if __name__ == "__main__":
    main()