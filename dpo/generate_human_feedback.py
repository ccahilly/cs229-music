import os
import pandas as pd
import simpleaudio as sa
import json
from generate_pairs import NUM_PAIRS_PER_CAPTION_PER_TEMP, TEMPS, caption_file, REF_IDX, POL_IDX
from convert_audio import converted_audio_dir

# Configurations
feedback_file = "../data/dpo-gen/human_labels_temp.json"

# Helper function to find the label index for a given ytid and pair_idx
def find_label_index(ytid, pair_idx, temp, labels):
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

def collect_input_for_pair(ytid, pair_idx, temp, labels, captions_df, stop_early=False):
    # Skip if label already exists and preference is set
    existing_label_idx = find_label_index(ytid, pair_idx, temp, labels)
    if existing_label_idx != -1 and labels[existing_label_idx]["preference"] != -1:
        return
    
    ref_file = os.path.join(converted_audio_dir, f"{ytid}-temp{temp}-pair{pair_idx}-{REF_IDX}.wav")
    pol_file = os.path.join(converted_audio_dir, f"{ytid}-temp{temp}-pair{pair_idx}-{POL_IDX}.wav")
    
    # Skip if files are missing
    if not os.path.exists(ref_file) or not os.path.exists(pol_file):
        print(f"Missing files for {ytid} pair {pair_idx}. Skipping.")
        return

    # Display caption
    caption = captions_df[captions_df["ytid"] == ytid]["caption"].values[0]
    print(f"\nCaption: {caption}\n")
    print(f"1. {ref_file}")
    print(f"2. {pol_file}}\n")

    # Playback loop
    while True:
        print("Type '1' to play first audio, '2' to play second audio, or 'n' to move to rating.")
        choice = input("Play option (1/2/n): ").strip()
        if choice == "1":
            play_audio(ref_file)
        elif choice == "2":
            play_audio(pol_file)
        elif choice == "n":
            break
        else:
            print("Invalid input. Please try again.")

    # Collect human preference
    while True:
        preference = input("Enter your preference (1 for first, 2 for second, 0 to skip, or q to quit): ").strip()
        if preference in {"1", "2", "0"}:
            if preference == "0":
                break
            elif preference == "1":
                p = REF_IDX
            else:
                p = POL_IDX # 2
            
            labels.append({"ytid": ytid, "pair_idx": pair_idx, "temp": temp, "preference": p})
            break
        elif preference == "q":
            stop_early = True
            break
        else:
            print("Invalid input. Please enter 1, 2, 0, or q.")

    return stop_early

def main():
    print("Note that this must be run locally for the sound to work.")
    # print("You can click 'c' while a song is playing to stop it.")

    # Load captions
    captions_df = pd.read_csv(caption_file)

    # Initialize labels
    if os.path.exists(feedback_file):
        with open(feedback_file, "r") as f:
            labels = json.load(f)
    else:
        labels = []

    for f in os.listdir(converted_audio_dir):
        # Extract ytid and check if it's valid
        if not f.endswith(".wav"):
            continue
        ytid = "-".join(f.split("-")[:-3])
        if ytid == "":
            # print(f"Invalid file name: {f}")
            continue

        for temp in TEMPS:
            for pair_idx in range(NUM_PAIRS_PER_CAPTION_PER_TEMP):
                stop_early = collect_input_for_pair(ytid, pair_idx, temp, labels, captions_df)
                
                if stop_early:
                    break
            
            if stop_early:
                break
        
        if stop_early:
            break

    # Save labeled data
    with open(feedback_file, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Human labels saved to {feedback_file}")

if __name__ == "__main__":
    main()