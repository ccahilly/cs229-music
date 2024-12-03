import os
import pandas as pd
import simpleaudio as sa
import json
from generate_pairs import NUM_PAIRS_PER_CAPTION_PER_TEMP, TEMPS

# Configurations
converted_audio_dir = "../data/dpo-gen-output-converted/"
caption_file = "../data/musiccaps-train-data.csv"
output_file = "../data/dpo-gen-output/human_labels_temp.json"

print("Note that this must be run locally for the sound to work.")
# print("You can click 'c' while a song is playing to stop it.")

# Load captions
captions_df = pd.read_csv(caption_file)

# Initialize labels
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        labels = json.load(f)
else:
    labels = []

# Helper function to find the label index for a given ytid and pair_idx
def find_label_index(ytid, pair_idx, temp):
    for i, label in enumerate(labels):
        if label["ytid"] == ytid and label["pair_idx"] == pair_idx and label["temp"] == temp:
            return i
    return -1

# Helper function to play an audio file
def play_audio(file_path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing audio: {e}")

stop_early = False
for f in os.listdir(converted_audio_dir):
    # Extract ytid and check if it's valid
    print(f)
    if (not f.endswith(".wav")) or (not "temp" not in f):
        continue
    ytid = "-".join(f.split("-")[:-3])
    if ytid == "":
        # print(f"Invalid file name: {f}")
        continue
    else:
        print(f"{ytid}")
    continue

    for temp in TEMPS:
        for pair_idx in range(NUM_PAIRS_PER_CAPTION_PER_TEMP):
            # Skip if label already exists and preference is set
            existing_label_idx = find_label_index(ytid, pair_idx, temp)
            if existing_label_idx != -1 and labels[existing_label_idx]["preference"] != -1:
                continue
            
            file_0 = os.path.join(converted_audio_dir, f"{ytid}-temp{temp}-pair{pair_idx}-0.wav")
            file_1 = os.path.join(converted_audio_dir, f"{ytid}-temp{temp}-pair{pair_idx}-1.wav")
            
            # Skip if files are missing
            if not os.path.exists(file_0) or not os.path.exists(file_1):
                print(f"Missing files for {ytid} pair {pair_idx}. Skipping.")
                continue

            # Display caption
            caption = captions_df[captions_df["ytid"] == ytid]["caption"].values[0]
            print(f"\nCaption: {caption}\n")
            # print(f"Pair {pair_idx}:")
            print(f"1. {file_0}")
            print(f"2. {file_1}\n")

            # Playback loop
            while True:
                print("Type '1' to play first audio, '2' to play second audio, or 'n' to move to rating.")
                choice = input("Play option (1/2/n): ").strip()
                if choice == "1":
                    play_audio(file_0)
                elif choice == "2":
                    play_audio(file_1)
                elif choice == "n":
                    break
                else:
                    print("Invalid input. Please try again.")

            # Collect human preference
            while True:
                preference = input("Enter your preference (1 for first, 2 for second, 0 to skip, or q to quit): ").strip()
                if preference in {"1", "2", "0"}:
                    if existing_label_idx != -1:
                        labels[existing_label_idx]["preference"] = int(preference) - 1
                    else:
                        labels.append({"ytid": ytid, "pair_idx": pair_idx, "temp": temp, "preference": int(preference) - 1})
                    break
                elif preference == "q":
                    stop_early = True
                    break
                else:
                    print("Invalid input. Please enter 0, 1, or -1.")
            
            if stop_early:
                break
        
        if stop_early:
            break

    if stop_early:
        break

# Save labeled data
with open(output_file, "w") as f:
    json.dump(labels, f, indent=2)

print(f"Human labels saved to {output_file}")
